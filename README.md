# ⚡ ComfyUI Qwen-i2L (Optimized Edition)

A fully optimized ComfyUI custom node for **Qwen-Image-i2L** (Image-to-LoRA). 
This node allows you to "extract" style, composition, or details from any image and save them as a lightweight **LoRA** (`.safetensors`) file.

**Key Feature:** It converts the heavy 24GB+ VRAM requirement of the official pipeline into a stable **8GB VRAM** workflow using smart offloading and FP8 compression.

---

## 🚀 Key Features

* **🧠 Smart VRAM Detection:** Automatically detects your GPU VRAM.
    * **< 16GB VRAM:** Automatically runs the heavy DINOv3 encoder on **CPU (RAM)** to prevent crashes.
    * **> 16GB VRAM:** Automatically uses **GPU** for maximum speed.
* **📉 FP8 Model Support:** Built to work with highly compressed **FP8 models** (7GB vs 14GB), saving 50% VRAM with zero quality loss.
* **💾 Smart Auto-Download:** Automatically pulls optimized models from the [markasd/QWEN_I2L](https://huggingface.co/markasd/QWEN_I2L) Hugging Face repository, so you don't need to manually hunt for files.
* **🧹 VRAM Nuke:** Features a built-in safety mechanism that aggressively clears ComfyUI's VRAM cache before processing to ensure stability.
* **✨ One-Click LoRA:** Takes an image input and outputs a ready-to-use LoRA file.

---

## 📦 Installation

### **Method 1: ComfyUI Manager (Easy)**
1.  Open ComfyUI Manager.
2.  Click **"Install via Git URL"**.
3.  Paste this link: 
    ```
    [https://github.com/gajjar4/ComfyUI-Qwen-Image-i2L](https://github.com/gajjar4/ComfyUI-Qwen-Image-i2L)
    ```
4.  Click Install and Restart ComfyUI.

### **Method 2: Manual Installation**
1.  Navigate to your custom nodes folder:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone this repository:
    ```bash
    git clone [https://github.com/gajjar4/ComfyUI-Qwen-Image-i2L.git](https://github.com/gajjar4/ComfyUI-Qwen-Image-i2L.git)
    ```
3.  Install dependencies:
    ```bash
    cd ComfyUI-Qwen-Image-i2L
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

---

## 🛠️ How It Works (The Pipeline)

The generation process uses a complex chain of Vision Encoders and Adapters to "understand" your image:

1.  **SigLIP (Vision Encoder):** "Looks" at the image to understand general semantics.
2.  **DINOv3 (Vision Encoder):** The heavy lifter. It analyzes the deep structure, composition, and high-level details. 
    * *Optimization:* This node intelligently offloads this 7GB giant to your System RAM if you have a small GPU (8GB/12GB cards).
3.  **Qwen-i2L (Adapter):** Takes the vision data and calculates how to shift the Diffusion Model's weights to match the style.
4.  **LoRA Generator:** Instead of retraining, it instantly saves these weight shifts as a `.safetensors` LoRA.

---

## 🎛️ Usage Guide

### **Step 1: The "Factory" Workflow (Creating the LoRA)**
*Use this workflow only to CREATE the file.*

1.  **Add Node:** `Qwen Pipeline Loader`
    * **Preset:** Choose `Style` (Art), `Coarse` (Shapes/Composition), or `Fine` (Details).
    * *Note: First run will auto-download models (~7GB) to `ComfyUI/models/I2L`.*
2.  **Add Node:** `Qwen i2L Apply`
    * Connect `qwen_file_path` from the Loader.
    * Connect your **Style Image** to `images`.
    * **Model Input:** Leave EMPTY (Disconnect it) to save VRAM.
3.  **Add Node:** `Save LoRA`
    * Connect `lora_data` from the Apply node.
    * Set your filename (e.g., `my_cool_style`).
4.  **Queue Prompt:** Wait for the analysis to finish. It will save the file to `ComfyUI/output/loras/`.

### **Step 2: The "Generation" Workflow (Using the LoRA)**
*Once you have the file, you don't need the Qwen nodes anymore.*

1.  Load your standard Text-to-Image workflow (Flux, SDXL, etc.).
2.  Add a standard **Load LoRA** node.
3.  Select the `my_cool_style.safetensors` you just created.
4.  Generate images!

---

## 📂 Folder Structure
The node automatically manages models in your ComfyUI directory:
`ComfyUI/models/I2L/`

* `/CLIPS/`: Stores SigLIP and DINOv3 (FP8).
* `/LORA models/`: Stores Qwen Adapters (Coarse, Fine, Style).

---

## 🤝 Credits
* **Code & Optimization:** gajjar4
* **Original Pipeline:** [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* **Optimized FP8 Models:** Hosted by `markasd`
