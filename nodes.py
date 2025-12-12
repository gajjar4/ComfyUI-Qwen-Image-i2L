import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import folder_paths
from safetensors.torch import save_file
import comfy.model_management
from transformers import AutoProcessor

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

class ForceCPUWrapper(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module
    
    def forward(self, x, *args, **kwargs):
        # Force input to CPU to avoid "Meta device" errors
        if hasattr(x, "to") and x.device.type != "cpu":
            x = x.to("cpu")
        return self.original_module(x, *args, **kwargs)

class QwenI2L_PipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset_mode": (["Style", "Coarse", "Fine", "Bias"],),
            }
        }

    RETURN_TYPES = ("QWEN_FILE",)
    RETURN_NAMES = ("qwen_file_path",)
    FUNCTION = "load_local"
    CATEGORY = "Qwen_i2L"

    def load_local(self, preset_mode):
        base_path = folder_paths.models_dir
        base_dir = os.path.join(base_path, "I2L")
        
        lora_dir = os.path.join(base_dir, "LORA models")
        clips_dir = os.path.join(base_dir, "CLIPS")

        print(f"\n{'='*60}")
        print(f"🔧 Qwen i2L Pipeline Loader - Mode: {preset_mode}")
        print(f"📁 Base Directory: {base_dir}")
        print(f"{'='*60}\n")

        # Map preset mode to filename
        filename_map = {
            "Bias": "Qwen-Image-i2L-Bias.safetensors",
            "Coarse": "Qwen-Image-i2L-Coarse.safetensors",
            "Fine": "Qwen-Image-i2L-Fine.safetensors",
            "Style": "Qwen-Image-i2L-Style.safetensors"
        }
        
        filename = filename_map[preset_mode]
        i2l_path = os.path.join(lora_dir, filename)
        
        # Check required files
        siglip_path = os.path.join(clips_dir, "SigLIP2-G384.safetensors")
        dino_path = os.path.join(clips_dir, "DINOv3-7B.safetensors")

        print("📋 Checking Required Files:\n")
        
        files_to_check = [
            ("i2L Model", i2l_path),
            ("SigLIP2-G384", siglip_path),
            ("DINOv3-7B", dino_path),
        ]
        
        missing_files = []
        for name, path in files_to_check:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✅ {name}: {size_mb:.1f} MB")
            else:
                print(f"  ❌ {name}: NOT FOUND")
                missing_files.append((name, path))
        
        if missing_files:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Missing {len(missing_files)} file(s)")
            print(f"{'='*60}\n")
            print("Please place the following files in the correct locations:\n")
            for name, path in missing_files:
                print(f"  • {name}")
                print(f"    Expected: {path}\n")
            print(f"{'='*60}\n")
            raise FileNotFoundError("Required model files are missing. See console output above.")

        print(f"\n{'='*60}")
        print(f"✅ All Required Files Found!")
        print(f"{'='*60}\n")

        return (i2l_path,)

class QwenI2L_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_file_path": ("QWEN_FILE",),
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "use_cpu_for_vision": ("BOOLEAN", {"default": True, "label": "Force CPU for Vision Encoders (Safe for 8GB VRAM)"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),  # Optional GGUF CLIP input
            }
        }

    RETURN_TYPES = ("MODEL", "lora_weights") 
    RETURN_NAMES = ("patched_model", "lora_data")
    FUNCTION = "apply_style"
    CATEGORY = "Qwen_i2L"

    def apply_style(self, qwen_file_path, images, strength, use_cpu_for_vision, model=None, clip=None):
        if not DIFFSYNTH_AVAILABLE: 
            raise Exception("❌ DiffSynth-Studio not installed! Run: pip install diffsynth")

        # Clear memory
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()

        # Check if this is a non-Style model (needs text encoder)
        needs_text_encoder = "Style" not in os.path.basename(qwen_file_path)

        device_str = "cpu" if use_cpu_for_vision else "cuda"
        mode_name = "SAFE (CPU)" if use_cpu_for_vision else "FAST (GPU)"
        
        print(f"\n{'='*60}")
        print(f"🎨 Qwen i2L Apply - {mode_name}")
        print(f"📊 Strength: {strength}")
        print(f"🖼️ Images: {len(images)}")
        if needs_text_encoder:
            print(f"⚠️ Non-Style model - Text encoder required")
        print(f"{'='*60}\n")
        
        base_path = folder_paths.models_dir
        clips_dir = os.path.join(base_path, "I2L", "CLIPS")
        
        siglip_path = os.path.join(clips_dir, "SigLIP2-G384.safetensors")
        dino_path = os.path.join(clips_dir, "DINOv3-7B.safetensors")

        # ---------------------------------------------------------
        # THE FIX: Pure RAM Mode (offload_device="cpu")
        # "disk" causes 'Meta Device' crash. "cpu" is safe for 32GB RAM.
        # ---------------------------------------------------------
        active_config = {
            "offload_dtype": torch.bfloat16, 
            "offload_device": "cpu", # <--- CHANGED FROM 'disk' to 'cpu'
            "onload_dtype": torch.bfloat16, 
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16, 
            "preparing_device": device_str,
            "computation_dtype": torch.bfloat16, 
            "computation_device": device_str,
        }

        # Convert ComfyUI images to PIL
        pil_images = []
        for img in images:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        try:
            print("\n📄 Loading Pipeline...")
            
            # Build model configs list
            model_configs = [
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern=siglip_path, 
                    **active_config
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern=dino_path, 
                    **active_config
                ),
                # Add i2L model
                ModelConfig(
                    model_id="DiffSynth-Studio/Qwen-Image-i2L", 
                    origin_file_pattern=qwen_file_path, 
                    **active_config
                )
            ]
            
            pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device_str, 
                model_configs=model_configs,
                processor_config=ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit",
                    origin_file_pattern="processor/"
                ),
            )
            
            print("✅ Pipeline Loaded Successfully\n")
            
            # If external CLIP provided, use it as text encoder (YOUR LOGIC)
            if needs_text_encoder:
                if clip is not None:
                    print("🔗 Connecting GGUF CLIP to pipeline...")
                    try:
                        # Extract the actual model from ComfyUI CLIP wrapper
                        if hasattr(clip, 'cond_stage_model'):
                            pipe.text_encoder = clip.cond_stage_model
                        elif hasattr(clip, 'load_model'):
                            # GGUF CLIP from city96's loader
                            pipe.text_encoder = clip.load_model()
                        else:
                            # Try to use it directly
                            pipe.text_encoder = clip
                        
                        print("✅ GGUF CLIP connected successfully\n")
                    except Exception as e:
                        print(f"❌ Failed to connect GGUF CLIP: {e}")
                else:
                    print("⚠️ WARNING: No CLIP connected! 'Fine'/'Coarse' modes WILL CRASH.")

            # Apply CPU wrappers if needed
            if use_cpu_for_vision:
                print("🛡️ Applying CPU wrappers to vision encoders...")
                pipe.siglip2_image_encoder = ForceCPUWrapper(pipe.siglip2_image_encoder)
                pipe.dinov3_image_encoder = ForceCPUWrapper(pipe.dinov3_image_encoder)
                print("✅ CPU wrappers applied\n")

            print("🔍 Analyzing images...")
            with torch.no_grad():
                # Encode images
                embs = QwenImageUnit_Image2LoRAEncode().process(
                    pipe, 
                    image2lora_images=pil_images
                )
                print("✅ Images encoded\n")
                
                # Ensure tensors are on correct device (The Bridge)
                forced_embs = {}
                for k, v in embs.items():
                    if isinstance(v, torch.Tensor):
                        forced_embs[k] = v.to(device_str)
                    else:
                        forced_embs[k] = v

                # Decode to LoRA weights
                print("⚙️ Generating LoRA weights...")
                result = QwenImageUnit_Image2LoRADecode().process(pipe, **forced_embs)
                lora_weights = result["lora"]
                print("✅ LoRA weights generated\n")

            # Apply strength multiplier
            if strength != 1.0:
                print(f"📊 Applying strength multiplier: {strength}")
                for k in lora_weights: 
                    lora_weights[k] *= strength
            
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS: LoRA Generated!")
            print(f"📦 Contains {len(lora_weights)} weight tensors")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Inference Failed")
            print(f"{'='*60}")
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()
            lora_weights = {}

        # Cleanup
        if 'pipe' in locals():
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
        if not lora_data:
            print("⚠️ No LoRA data to save (inference failed)")
            return ()
        
        # Remove .safetensors if user added it
        if filename.endswith('.safetensors'):
            filename = filename[:-12]
            
        out_dir = os.path.join(folder_paths.get_output_directory(), "loras")
        os.makedirs(out_dir, exist_ok=True)
        
        save_path = os.path.join(out_dir, f"{filename}.safetensors")
        
        try:
            save_file(lora_data, save_path)
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            
            print(f"\n{'='*60}")
            print(f"💾 LoRA Saved Successfully!")
            print(f"📁 Path: {save_path}")
            print(f"📊 Size: {file_size:.2f} MB")
            print(f"📦 Tensors: {len(lora_data)}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"❌ Failed to save LoRA: {e}")
        
        return ()

# Node mappings
NODE_CLASS_MAPPINGS = {
    "QwenI2L_PipelineLoader": QwenI2L_PipelineLoader,
    "QwenI2L_Apply": QwenI2L_Apply,
    "QwenI2L_Save": QwenI2L_Save
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenI2L_PipelineLoader": "🔧 Qwen i2L Loader",
    "QwenI2L_Apply": "🎨 Qwen i2L Apply",
    "QwenI2L_Save": "💾 Save LoRA"
}
