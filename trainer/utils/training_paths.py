from pathlib import Path
import os
import trainer.constants as train_cst
from trainer.utils.style_detection import detect_styles_in_prompts
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import ImageModelType

def get_checkpoints_output_path(task_id: str, repo_name: str) -> str:
    return str(Path(train_cst.OUTPUT_CHECKPOINTS_PATH) / task_id / repo_name)

def get_image_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    base_path = str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)
    if os.path.isdir(base_path):
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(base_path, files[0])
    return base_path

def get_image_training_images_dir(task_id: str) -> str:
    return str(Path(train_cst.IMAGE_CONTAINER_IMAGES_PATH) / task_id / "img")

def get_image_training_config_template_path(model_type: str, train_data_dir: str) -> tuple[str, bool]:
    model_type = model_type.lower()
    if model_type == ImageModelType.SDXL.value:
        # Find the first subdirectory in train_data_dir that contains text files
        prompts = []
        found_prompts_dir = False
        
        if os.path.exists(train_data_dir):
            for item in os.listdir(train_data_dir):
                item_path = os.path.join(train_data_dir, item)
                if os.path.isdir(item_path):
                    # Check for .txt files in this directory
                    txt_files = [f for f in os.listdir(item_path) if f.endswith(".txt")]
                    if txt_files:
                        found_prompts_dir = True
                        for file in txt_files:
                            try:
                                with open(os.path.join(item_path, file), "r") as f:
                                    prompt = f.read().strip()
                                    prompts.append(prompt)
                            except Exception as e:
                                print(f"Error reading prompt file {file}: {e}")
                        # We assume all prompts are in one directory for now, or we collect from all valid directories.
                        # The original code only looked at one specific folder. 
                        # To be safe and likely match the intent of 'prepare_dataset' which creates one folder, we can break after finding one.
                        break 
        
        if not found_prompts_dir:
             print(f"Warning: No directory with .txt files found in {train_data_dir}. Using default person config.")
             return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_sdxl_person.toml"), False

        styles = detect_styles_in_prompts(prompts)
        print(f"Styles: {styles}")

        if styles:
            return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_sdxl_style.toml"), True
        else:
            return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_sdxl_person.toml"), False

    elif model_type == ImageModelType.FLUX.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_flux.toml"), False

def get_image_training_zip_save_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_tourn.zip")

def get_text_dataset_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_train_data.json")

def get_axolotl_dataset_paths(dataset_filename: str) -> tuple[str, str]:
    data_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["data"]) / dataset_filename)
    root_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["root"]) / dataset_filename)
    return data_path, root_path

def get_axolotl_base_config_path(dataset_type) -> str:
    root_dir = Path(train_cst.AXOLOTL_DIRECTORIES["root"])
    if isinstance(dataset_type, (InstructTextDatasetType, DpoDatasetType)):
        return str(root_dir / "base.yml")
    elif isinstance(dataset_type, GrpoDatasetType):
        return str(root_dir / "base_grpo.yml")
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")

def get_text_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    return str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)