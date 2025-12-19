
import sys
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# --- MOCKING START ---
# Mock core.models.utility_models to avoid pydantic dependency
mock_core = MagicMock()
sys.modules["core"] = mock_core
mock_models = MagicMock()
sys.modules["core.models"] = mock_models
mock_utility_models = MagicMock()
sys.modules["core.models.utility_models"] = mock_utility_models

# Define ImageModelType enum-like class on the mock
class ImageModelType:
    SDXL = MagicMock(value="sdxl")
    FLUX = MagicMock(value="flux")

mock_utility_models.ImageModelType = ImageModelType
mock_utility_models.DpoDatasetType = MagicMock()
mock_utility_models.GrpoDatasetType = MagicMock()
mock_utility_models.InstructTextDatasetType = MagicMock()
mock_utility_models.ChatTemplateDatasetType = MagicMock()

# --- MOCKING END ---

# Now we can import the module under test
# We need 'trainer' to be in path. Assuming we run from root.
import trainer.constants as train_cst
from trainer.utils.training_paths import get_image_training_config_template_path

# Override constants for testing
train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH = "/tmp/mock_templates"
os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH, exist_ok=True)

# Create dummy templates
with open(os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH, "base_diffusion_sdxl_style.toml"), "w") as f:
    f.write("style")
with open(os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH, "base_diffusion_sdxl_person.toml"), "w") as f:
    f.write("person")

def test_dynamic_path_discovery():
    base_test_dir = "/tmp/test_training_paths"
    if os.path.exists(base_test_dir):
        shutil.rmtree(base_test_dir)
    os.makedirs(base_test_dir)

    # Case 1: Directory with unpredictable name containing txt files (e.g., "10_lora style")
    case1_dir = os.path.join(base_test_dir, "case1")
    os.makedirs(case1_dir)
    random_folder_name = "10_lora_style_random"
    prompt_dir = os.path.join(case1_dir, random_folder_name)
    os.makedirs(prompt_dir)
    
    with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
        f.write("a photo of a person") 
        
    print(f"Testing with directory: {prompt_dir}")
    
    config_path, is_style = get_image_training_config_template_path("sdxl", case1_dir)
    print(f"Result Case 1: Config Path: {config_path}, Is Style: {is_style}")
    
    # Validation
    if "base_diffusion_sdxl_person.toml" in config_path and is_style is False:
        print("PASS: Case 1")
    else:
        print(f"FAIL: Case 1. Expected person config, got {config_path}, is_style={is_style}")

    # Case 2: Style prompt
    # Since we can't easily control detect_styles_in_prompts output without mocking it too (it's imported inside training_paths module scope?),
    # actually it is imported as `from trainer.utils.style_detection import detect_styles_in_prompts`
    # We rely on specific keywords in `style_detection.py`. "watercolor painting" is a style.
    
    case2_dir = os.path.join(base_test_dir, "case2")
    os.makedirs(case2_dir)
    style_folder = "style_folder"
    style_prompt_dir = os.path.join(case2_dir, style_folder)
    os.makedirs(style_prompt_dir)
    
    with open(os.path.join(style_prompt_dir, "prompt.txt"), "w") as f:
        f.write("a watercolor painting of a dog") 
        
    print(f"Testing with directory: {style_prompt_dir}")
    
    config_path, is_style = get_image_training_config_template_path("sdxl", case2_dir)
    print(f"Result Case 2: Config Path: {config_path}, Is Style: {is_style}")
    
    if "base_diffusion_sdxl_style.toml" in config_path and is_style is True:
        print("PASS: Case 2")
    else:
        print(f"FAIL: Case 2. Expected style config, got {config_path}, is_style={is_style}")

    # Case 3: Empty directory (should handle gracefully?)
    case3_dir = os.path.join(base_test_dir, "case3")
    os.makedirs(case3_dir)
    print(f"Testing with empty directory: {case3_dir}")
    
    config_path, is_style = get_image_training_config_template_path("sdxl", case3_dir)
    print(f"Result Case 3: Config Path: {config_path}, Is Style: {is_style}")
    if "base_diffusion_sdxl_person.toml" in config_path:
        print("PASS: Case 3")
    else:
        print("FAIL: Case 3")


if __name__ == "__main__":
    test_dynamic_path_discovery()
