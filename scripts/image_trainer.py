#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
OPTIMIZED STRATEGY: Rank Inflation + Prodigy Auto-Adapt + Universal Config Mapping
"""

import argparse
import asyncio
import os
import subprocess
import sys
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

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, repeats):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    # --- CONFIG MAPPING STRATEGY (RANK INFLATION) ---
    # Strategy: Upgrade specs (Rank/Conv) per tier to outperform standard baselines.
    
    # Database Model-to-Config (Original Champion Mapping reference)
    network_config_person = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 467,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 467,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
        "John6666/nova-anime-xl-pony-v5-sdxl": 240, # OPTIMIZED: Rank 64 (Sniper Mode)
        "cagliostrolab/animagine-xl-4.0": 699,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 467,
        "dataautogpt3/TempestV0.1": 456,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 467,
        "fluently/Fluently-XL-Final": 228,
        "mann-e/Mann-E_Dreams": 456,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 228,
        "recoilme/colorfulxl": 228,
        "zenless-lab/sdxl-aam-xl-anime-mix": 456,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
        "zenless-lab/sdxl-anything-xl": 228,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
        "Corcelio/mobius": 228,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
    }

    network_config_style = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 235,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 235,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 235,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 235,
        "dataautogpt3/TempestV0.1": 228,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 235,
        "fluently/Fluently-XL-Final": 235,
        "mann-e/Mann-E_Dreams": 235,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 235,
        "recoilme/colorfulxl": 235,
        "zenless-lab/sdxl-aam-xl-anime-mix": 235,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
        "zenless-lab/sdxl-anything-xl": 235,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
        "Corcelio/mobius": 235,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
    }

    # RE-ENGINEERED CONFIG MAPPING (STRATEGY: TRUE CLONE - RANK 32)
    config_mapping = {
        # TIER 1: Model Ringan/Art (Original: Rank 32)
        # STRATEGI: Rank 32 (Match Champion) + Alpha 1.
        # Rank 64 terbukti terlalu tidak stabil. Balik ke 32 untuk konsistensi.
        228: {
            "network_dim": 32,          # MATCH CHAMPION: 32
            "network_alpha": 1,         # PRODIGY MAGIC: 1
            "network_args": []
        },
        235: {
            "network_dim": 32,          # MATCH CHAMPION: 32
            "network_alpha": 32,        # MATCH CHAMPION: 32
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
        },
        # --- (SNIPER REVERTED TO TIER 1 SPEC) ---
        240: {
            "network_dim": 32,          # REVERT: Rank 32 (Stability King)
            "network_alpha": 1,         # KEEP: Alpha 1 (Prodigy Optimized)
            "network_args": ["conv_dim=4", "conv_alpha=1", "dropout=null"]
        },

        # TIER 2: Model Realis/Menengah (Original: Rank 64)
        # STRATEGI: Rank 128 + Alpha 1.
        # Menangkap pori-pori & tekstur kulit realistis dengan auto-scale max.
        456: {
            "network_dim": 128,         # UPGRADED from 64
            "network_alpha": 1,         # PRODIGY MAGIC: 1
            "network_args": []
        },
        467: {
            "network_dim": 128,         # UPGRADED from 64
            "network_alpha": 1,         # PRODIGY MAGIC: 1
            "network_args": ["conv_dim=8", "conv_alpha=1", "dropout=null"]
        },

        # TIER 3: Model Berat/Complex (Original: Rank 96)
        # STRATEGI: Upgrade ke Rank 160. Kapasitas masif untuk model raksasa (Animagine).
        699: {
            "network_dim": 160,         # UPGRADED from 96
            "network_alpha": 80,        # Stabil (Tier 3 experimental, keep safe high alpha)
            "network_args": ["conv_dim=8", "conv_alpha=4", "dropout=null"]
        },
    }

    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = train_data_dir
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    if model_type == "sdxl":
        if is_style:
            # STYLE TASK: Gunakan mapping yang sudah di-upgrade (Rank 64)
            # Fallback ke 235 jika model tidak dikenal
            network_config = config_mapping[network_config_style.get(model_name, 235)]
        else:
            # PERSON TASK: Gunakan mapping yang sudah di-upgrade (Rank 128/160)
            network_config = config_mapping[network_config_person.get(model_name, 235)]
            
            # STRATEGI TAMBAHAN: AUTO-OPTIMIZER
            # Paksa Optimizer Prodigy untuk Person Task
            # Karena kita pakai Rank Tinggi (128/160), kita butuh optimizer cerdas.
            print("💎 PERSON MODE DETECTED: Upgrading Optimizer to Prodigy for High-Rank Stability")
            config["optimizer_type"] = "prodigy"
            config["optimizer_args"] = ["decouple=True", "d_coef=1", "weight_decay=0.01", "use_bias_correction=True", "safeguard_warmup=True"]
            config["unet_lr"] = 1.0
            config["text_encoder_lr"] = 1.0

        config["network_dim"] = network_config["network_dim"]
        config["network_alpha"] = network_config["network_alpha"]
        config["network_args"] = network_config["network_args"]

    # --- EPOCH CALCULATION STRATEGY ---
    # FORCE TIER 1 EPOCH RULES:
    # - STYLE TASK: 25 Epochs (Pelan & Halus)
    # - PERSON TASK: 30 Epochs (Hafalan Kuat)
    
    # Deteksi Task Type dari Config yang sudah dibuat
    if is_style:
        target_epochs = 25
        print(f"🎨 Style Task Detected. FORCING {target_epochs} Epochs (Tier 1 Spec).")
    else:
        # Person Task
        num_images = len([f for f in os.listdir(train_data_dir) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".webp")])
        if num_images < 50:
            target_epochs = 30 # Micro/Small Person Dataset
            print(f"🧔 Micro/Small Person Dataset Detected. FORCING {target_epochs} Epochs (Tier 1 Spec).")
        else:
            target_epochs = 30 # Fallback Safe Person
            
    config["max_train_epochs"] = target_epochs
    config["save_every_n_epochs"] = 10 # Save less frequently to save disk space

    # Save config to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path} with {target_epochs} Epochs", flush=True)
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
        # Menangani cmd yang bisa berupa list atau string
        cmd_str = ' '.join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)
        print(f"Command: {cmd_str}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT (OPTIMIZED RANK INFLATION)---", flush=True)
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
    repeats = cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
    
    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=repeats,
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
        repeats
    )

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
