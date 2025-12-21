#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import os
import subprocess
import sys
import json

import toml


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


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path

def create_config(task_id, model_path, model_name, model_type, expected_repo_name):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    # --- 1. SETTING BASE YANG KUAT (UNIVERSAL) ---
    # Kita set baseline yang stabil untuk semua model.
    # Tidak ada lagi diskriminasi berdasarkan nama model.
    # Rank 128 + Conv Dim 8 adalah "sweet spot" antara stabilitas dan detail.
    
    config["network_dim"] = 128
    config["network_alpha"] = 128
    config["network_args"] = ["conv_dim=8", "conv_alpha=8", "dropout=null"]
    
    # Setup path standar
    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = train_data_dir
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # --- 2. LOAD SPECIAL CONFIG FROM JSON (The Missing Link) ---
    # Kita cari file json config yang sesuai (person atau style)
    # Lokasi file json di folder scripts/lrs/
    
    # Fix Path: Gunakan absolute path dari lokasi file script ini berada
    current_script_path = os.path.abspath(__file__) # c:/56/goy/image-test/scripts/image_trainer.py
    scripts_dir_path = os.path.dirname(current_script_path) # c:/56/goy/image-test/scripts
    lrs_dir = os.path.join(scripts_dir_path, "lrs") # c:/56/goy/image-test/scripts/lrs

    json_config_filename = "style_config.json" if is_style else "person_config.json"
    json_config_path = os.path.join(lrs_dir, json_config_filename)
    
    print(f"[DEBUG] Checking for LRS Config at: {json_config_path}", flush=True)
    
    special_config = {}
    
    # Coba baca file JSON
    if os.path.exists(json_config_path):
        try:
            with open(json_config_path, 'r') as f:
                full_json = json.load(f)
            
            print(f"[DEBUG] LRS File Loaded. Checking Task ID: {task_id}", flush=True)
                
            # Cek apakah Task ID kita punya settingan khusus
            if task_id in full_json.get("data", {}):
                print(f"[SUCCESS] Found SPECIAL CONFIG for Task ID: {task_id}", flush=True)
                special_config = full_json["data"][task_id]
                print(f"[DEBUG] Content to overwrite: {special_config}", flush=True)
            else:
                print(f"[INFO] No special config found for {task_id}. Using defaults.", flush=True)
                
        except Exception as e:
            print(f"[WARNING] Failed to load JSON config: {e}", flush=True)
    else:
        print(f"[WARNING] LRS Config file NOT FOUND at: {json_config_path}", flush=True)

    # --- 3. OVERWRITE CONFIG ---
    # Jika ada special config, timpa settingan TOML
    # Ini kuncinya! LR, Epochs, Noise Offset dari JSON akan masuk ke sini.
    
    for key, value in special_config.items():
        if key in config:
            print(f"   -> Overwriting {key}: {config[key]} -> {value}", flush=True)
            config[key] = value
        else:
            # Jika key belum ada di toml (misal optimizer args yang kompleks), kita tambahkan
            print(f"   -> Adding new setting {key}: {value}", flush=True)
            config[key] = value

    # Save config to file
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
            f"/app/sd-script/{model_type}_train_network.py",
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
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
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

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
    )

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
