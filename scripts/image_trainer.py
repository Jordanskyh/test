#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
IMPROVED: Hybrid Strategy + Research Backed Optimizations (Min-SNR, Multires, Dynamic)
FINAL CHECK: Fixed logic overwrites for dormant boosters
"""

import argparse
import asyncio
import os
import subprocess
import sys
import toml
import math
import json

# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType

# --- STRATEGY CONFIGURATION ---
TARGET_STEPS_STYLE = 2500
TARGET_STEPS_PERSON = 1200
MIN_EPOCHS = 10
MAX_EPOCHS = 100

def get_image_count(dir_path):
    """Menghitung jumlah gambar valid di direktori training"""
    count = 0
    valid_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(valid_ext):
                count += 1
    return count if count > 0 else 1

def calculate_dynamic_epochs(num_images, batch_size, repeats, is_style):
    target = TARGET_STEPS_STYLE if is_style else TARGET_STEPS_PERSON
    steps_per_epoch = (num_images * repeats) / batch_size
    if steps_per_epoch < 1: steps_per_epoch = 1
    ideal_epochs = math.ceil(target / steps_per_epoch)
    final_epochs = max(MIN_EPOCHS, min(MAX_EPOCHS, ideal_epochs))
    
    print(f"--- DYNAMIC EPOCH CALCULATION ---")
    print(f"Mode: {'STYLE' if is_style else 'PERSON'}")
    print(f"Images: {num_images} | Repeats: {repeats} | Epochs: {final_epochs}")
    return final_epochs

def get_special_config_from_lrs(task_id, is_style, model_type):
    """Membaca konfigurasi rahasia dari file LRS JSON"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lrs_dir = os.path.join(script_dir, "lrs")
    
    if model_type == "flux":
        filename = "flux.json"
    else:
        filename = "style_config.json" if is_style else "person_config.json"
        
    json_path = os.path.join(lrs_dir, filename)
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if task_id in data.get("data", {}):
                    print(f"[LRS] FOUND special config for Task {task_id}")
                    return data["data"][task_id]
        except Exception as e:
            print(f"[LRS] Error reading JSON: {e}")
            
    print(f"[LRS] No special config for Task {task_id}. Using Dynamic Logic.")
    return None

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, repeats):
    train_data_dir = train_paths.get_image_training_images_dir(task_id)
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    # 1. BASELINE STRONG CONFIG
    # SDXL butuh kapasitas besar (Dim 160)
    if model_type == "sdxl":
        config["network_dim"] = 160
        config["network_alpha"] = 160
        config["network_args"] = ["conv_dim=8", "conv_alpha=8", "dropout=null"]
    
    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = train_data_dir
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # 2. LOAD LRS (Rahasia) - Priority 1
    special_config = get_special_config_from_lrs(task_id, is_style, model_type)
    
    if special_config:
        for key, value in special_config.items():
            config[key] = value
            print(f"[LRS] Applied: {key} = {value}")
            
        if "max_train_epochs" not in special_config and model_type == "sdxl":
             num_images = get_image_count(train_data_dir)
             batch_size = config.get("train_batch_size", 4)
             config["max_train_epochs"] = calculate_dynamic_epochs(num_images, batch_size, repeats, is_style)

    else:
        # Priority 2: Full Dynamic Logic (Task Baru)
        if model_type == "sdxl":
            num_images = get_image_count(train_data_dir)
            batch_size = config.get("train_batch_size", 4)
            config["max_train_epochs"] = calculate_dynamic_epochs(num_images, batch_size, repeats, is_style)

    # 3. RESEARCH BACKED BOOSTERS (SDXL ONLY)
    if model_type == "sdxl":
        # Booster A: Multires Noise (Texture Quality)
        # FIX: Force Multires over generic default (0.035), unless it's champion specific (0.0411)
        current_noise = config.get("noise_offset")
        is_champion_value = (current_noise == 0.0411)
        
        if not is_champion_value:
             config["noise_offset_type"] = "Multires"
             config["multires_noise_iterations"] = 6
             config["multires_noise_discount"] = 0.3
             # Remove legacy noise_offset to avoid conflicts
             config.pop("noise_offset", None)
             print("[RESEARCH] Multires Noise Activated (Overwriting generic default)")
        else:
             print("[LRS] Keeping champion specific noise offset (0.0411).")
            
        # Booster B: Min-SNR Gamma (Training Stability - CVPR 2024)
        # FIX: Standardization to 5
        if config.get("min_snr_gamma") != 5:
            config["min_snr_gamma"] = 5
            print("[RESEARCH] Min-SNR Gamma adjusted to 5 (Stability Booster)")

    # 4. FLUX OPTIMIZATION (FLORA PAPER)
    if model_type == "flux":
        # Adaptive Guidance: Turunkan guidance untuk style agar lebih kreatif/fleksibel
        if is_style:
            # Default TOML is 85.0. Paper suggests lower for styles.
            config["guidance_scale"] = 25.0
            print("[RESEARCH] Flux Guidance Scale lowered to 25.0 for Style Task (Flexibility)")
        else:
            # Person task stays high (fidelity)
            pass

    # Save to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    if model_type == "sdxl":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-scripts/{model_type}_train_network.py",
            "--config_file", config_path
        ]
    elif model_type == "flux":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-scripts/{model_type}_train_network.py",
            "--config_file", config_path
        ]
    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        # More robust error reporting
        print(f"Exit Code: {e.returncode}", flush=True)
        cmd_str = ' '.join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)
        print(f"Command: {cmd_str}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT (FINAL CHECKED)---", flush=True)
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    print("Preparing dataset...", flush=True)
    repeats = cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
    
    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=repeats,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        repeats
    )

    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
