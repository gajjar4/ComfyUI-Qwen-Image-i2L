import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import requests
from tqdm import tqdm
from safetensors.torch import save_file
import comfy.model_management

CUSTOM_REPO_ID = "markasd/QWEN_I2L" 
# --------------------------------------

try:
    from diffsynth.pipelines.qwen_image import (
        QwenImagePipeline, 
        ModelConfig, 
        QwenImageUnit_Image2LoRAEncode, 
        QwenImageUnit_Image2LoRADecode
    )
    DIFFSYNTH_AVAILABLE = True
except ImportError:
    DIFFSYNTH_AVAILABLE = False
    print("\n[CRITICAL WARNING] DiffSynth-Studio not found!\n")

def smart_download(url, save_path, token=None):
    if os.path.exists(save_path):
        if os.path.getsize(save_path) < 10 * 1024 * 1024: 
            try: os.remove(save_path)
            except: pass
        else:
            print(f"[Qwen] Found existing model: {save_path}")
            return True

    print(f"[Qwen] Downloading from Custom Repo ({CUSTOM_REPO_ID}) to: {save_path}")
    headers = {"Authorization": f"Bearer {token}"} if token and "Paste" not in token else {}
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False

class QwenI2L_PipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset_mode": (["Bias", "Coarse", "Fine", "Style"],),
                "hf_token": ("STRING", {"default": "Paste_Your_HF_Token_Here", "multiline": False}),
            }
        }

    RETURN_TYPES = ("QWEN_FILE",)
    RETURN_NAMES = ("qwen_file_path",)
    FUNCTION = "load_or_download"
    CATEGORY = "Qwen_i2L"

    def load_or_download(self, preset_mode, hf_token):
        base_path = folder_paths.models_dir
        target_dir = os.path.join(base_path, "I2L", "LORA models")
        os.makedirs(target_dir, exist_ok=True)

        filename_map = {
            "Bias": "Qwen-Image-i2L-Bias.safetensors",
            "Coarse": "Qwen-Image-i2L-Coarse.safetensors",
            "Fine": "Qwen-Image-i2L-Fine.safetensors",
            "Style": "Qwen-Image-i2L-Style.safetensors"
        }
        
        filename = filename_map[preset_mode]
        full_path = os.path.join(target_dir, filename)
        
        url = f"https://huggingface.co/{CUSTOM_REPO_ID}/resolve/main/{filename}"
        if not smart_download(url, full_path, hf_token): return ("",)

        clips_dir = os.path.join(base_path, "I2L", "CLIPS")
        os.makedirs(clips_dir, exist_ok=True)
        
        print("--- Checking Vision Encoders ---")
        
        smart_download(
            "https://huggingface.co/DiffSynth-Studio/General-Image-Encoders/resolve/main/SigLIP2-G384/model.safetensors", 
            os.path.join(clips_dir, "SigLIP2-G384.safetensors"), 
            hf_token
        )
        
        smart_download(
            f"https://huggingface.co/{CUSTOM_REPO_ID}/resolve/main/DINOv3-7B.safetensors", 
            os.path.join(clips_dir, "DINOv3-7B.safetensors"), 
            hf_token
        )

        return (full_path,)

class QwenI2L_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_file_path": ("QWEN_FILE",),
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            },
            "optional": {"model": ("MODEL",)}
        }

    RETURN_TYPES = ("MODEL", "lora_weights") 
    RETURN_NAMES = ("patched_model", "lora_data")
    FUNCTION = "apply_style"
    CATEGORY = "Qwen_i2L"

    def apply_style(self, qwen_file_path, images, strength, model=None):
        if not DIFFSYNTH_AVAILABLE: raise Exception("DiffSynth-Studio missing!")

        # 1. Clear VRAM
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()

        total_vram = comfy.model_management.get_total_memory(torch.device("cuda")) / (1024**3)
        
        use_cpu_for_dino = total_vram < 15.5 
        
        mode_str = "CPU (Safe Mode)" if use_cpu_for_dino else "GPU (Fast Mode)"
        print(f"--- Qwen i2L: Detected {total_vram:.1f} GB VRAM. Using {mode_str} for DINO. ---")
        
        base_path = folder_paths.models_dir
        clips_dir = os.path.join(base_path, "I2L", "CLIPS")
        siglip_path = os.path.join(clips_dir, "SigLIP2-G384.safetensors")
        dino_path = os.path.join(clips_dir, "DINOv3-7B.safetensors")

        gpu_config = {
            "offload_dtype": "disk", "offload_device": "disk", 
            "onload_dtype": torch.bfloat16, "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16, "computation_device": "cuda",
        }
        cpu_config = gpu_config.copy()
        cpu_config["preparing_device"] = "cpu"
        cpu_config["computation_device"] = "cpu"
        
        dino_config = cpu_config if use_cpu_for_dino else gpu_config

        pil_images = [Image.fromarray(np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)) for img in images]

        try:
            pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern=siglip_path, **gpu_config),
                    ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern=dino_path, **dino_config),
                    ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-i2L", origin_file_pattern=qwen_file_path, **gpu_config)
                ]
            )

            print("Analyzing images...")
            with torch.no_grad():
                embs = QwenImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=pil_images)
                result = QwenImageUnit_Image2LoRADecode().process(pipe, **embs)
                lora_weights = result["lora"]

            if strength != 1.0:
                for k in lora_weights: lora_weights[k] *= strength
            
            print(f"--- Qwen i2L: Success! LoRA Created. ---")

        except Exception as e:
            print(f"[ERROR] Inference Failed: {e}")
            lora_weights = {}

        del pipe
        torch.cuda.empty_cache()
        
        return (model.clone() if model else None, lora_weights)

class QwenI2L_Save:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_data": ("lora_weights",),
                "filename": ("STRING", {"default": "qwen_style_lora"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_lora"
    CATEGORY = "Qwen_i2L"

    def save_lora(self, lora_data, filename):
        out_dir = os.path.join(folder_paths.get_output_directory(), "loras")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{filename}.safetensors")
        save_file(lora_data, path)
        print(f"Saved LoRA to: {path}")
        return ()

NODE_CLASS_MAPPINGS = {
    "QwenI2L_PipelineLoader": QwenI2L_PipelineLoader,
    "QwenI2L_Apply": QwenI2L_Apply,
    "QwenI2L_Save": QwenI2L_Save
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenI2L_PipelineLoader": "1. Qwen Pipeline Loader",
    "QwenI2L_Apply": "2. Qwen i2L Apply",
    "QwenI2L_Save": "3. Save LoRA"
}