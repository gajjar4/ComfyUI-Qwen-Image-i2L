import subprocess
import sys

def install_package(package_name):
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")

if __name__ == "__main__":
    print("Installing dependencies for Qwen-Image-i2L...")
    install_package("diffsynth>=1.1.9")
    install_package("huggingface_hub")
    install_package("safetensors")
    install_package("modelscope") 
    
    print("\nInstallation complete. Please restart ComfyUI.")