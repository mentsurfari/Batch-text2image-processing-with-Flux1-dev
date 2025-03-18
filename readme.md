Markdown

# FLUX.1 Image Generation Script

This script generates images from text prompts using the FLUX.1 diffusion model. It loads the necessary model components (UNet, VAE, CLIP, T5) from local files and processes prompts from a CSV file to generate corresponding images.

## Features

* Loads UNet model weights from GGUF format.
* Loads VAE and CLIP model weights from safetensors format.
* Loads T5 model weights from GGUF format.
* Supports text prompts for image generation.
* Reads prompts from a CSV file for batch image generation.
* Saves generated images as PNG files.
* Includes basic error handling and progress display.
* Provides an "emergency decode" fallback in case the VAE decoder fails.
* Performs a basic test image generation to verify the setup.
* Checks for GPU availability and provides warnings if running on CPU.

## Dependencies

* Python 3.x
* `torch` (>= 2.0)
* `gguf`
* `diffusers`
* `transformers`
* `safetensors`
* `Pillow` (PIL)
* `numpy`
* `scikit-image` (for the improved emergency decode function)

You can install these dependencies using pip:

```bash
pip install torch gguf diffusers transformers safetensors Pillow numpy scikit-image
Note: For GPU support, ensure you have the correct CUDA toolkit and drivers installed for your system and the appropriate PyTorch version. Refer to the PyTorch website for installation instructions.

Setup
Clone the repository:

Bash

git clone <repository_url>
cd <repository_directory>
Download Model Files: This script expects the following model files to be available at the specified hardcoded paths. You will need to obtain these files separately.

UNET: flux1-dev-Q4_K_S.gguf (Expected path: C:/Users/userr/Documents/ComfyUI/models/diffusion_models/FLUX1/)
VAE: ae.safetensors (Expected path: C:/Users/userr/Documents/ComfyUI/models/vae/FLUX1/)
CLIP: clip_l.safetensors (Expected path: C:/Users/userr/Documents/ComfyUI/models/text_encoders/clip_l.safetensors)
T5 Encoder: t5-v1_1-xxl-encoder-Q4_K_S.gguf (Expected path: C:/Users/userr/Documents/ComfyUI/models/text_encoders/t5/)
CLIP Tokenizer: Located in C:/Users/userr/Documents/ComfyUI/models/clip_tokenizer/
T5 Tokenizer: Located in C:/Users/userr/Documents/ComfyUI/models/t5_tokenizer/
Important: You will need to adjust the file paths in the main() function of the script to match the actual locations of your model files if they are different.

Prepare CSV File: Create a CSV file containing the text prompts you want to use for image generation. The script, by default, reads the prompts from the second column (index 1) of the CSV file located at C:/Users/userr/Documents/GitHub/video-creation-automation/[hiphop]giftoftnothing_10_beat_clips/all_transcriptions.csv.

The first row of the CSV is expected to be a header and will be skipped.
Empty rows and rows with insufficient columns will be skipped with a warning.
Empty prompts will also be skipped.
Output Folder: The generated images will be saved in the generated_images folder within the same directory as the script, or at the path specified by output_folder in the main() function. This folder will be created if it doesn't exist.

Usage
Navigate to the script's directory:

Bash

cd <repository_directory>
Run the script:

Bash

python your_script_name.py
Replace your_script_name.py with the actual name of the Python file.

Follow the prompts: The script will first perform a GPU check and warn you if it's running on the CPU. It will then attempt to load the model files and perform a test image generation. You will be asked to verify the quality of the test image before proceeding with the full CSV processing.

Generated Images: Once the script processes the CSV file, the generated images will be saved as image_1.png, image_2.png, etc., in the specified output folder.

Configuration
The main configuration options are within the main() function of the script:

unet_path, vae_path, clip_path, t5_path: These variables define the file paths to the respective model files. You will likely need to modify these to match your local setup.
csv_path: Path to the CSV file containing the prompts.
output_folder: Path to the folder where generated images will be saved.
column_index (within the CSV reading logic): Determines which column in the CSV file contains the prompts (default is 1 for the second column).
Generation Parameters (within the generate_image function):
guidance_scale: Controls the strength of the guidance towards the prompt (default: 7.5). Higher values usually result in images that are more closely aligned with the prompt but might also introduce artifacts.
num_inference_steps: The number of denoising steps to perform (default: 30, increased to 40 in the CSV processing loop). More steps generally lead to higher quality but take longer.
width, height: The dimensions of the generated images (default: 512x512).
seed: A random seed for reproducibility. If not provided, a random seed will be used for each image.
You can adjust these parameters in the script to customize the image generation process.

Troubleshooting
CUDA Not Available: If the script reports that CUDA is not available, ensure you have a compatible NVIDIA GPU, the correct drivers installed, and a CUDA-enabled version of PyTorch.
Model File Not Found: Double-check the file paths in the main() function and ensure that the model files exist at those locations.
VAE Decoding Failed: The script includes a fallback "emergency decode" function, but the quality of images generated with this method might be lower. If the VAE consistently fails, ensure the VAE model file is correct and compatible with the FLUX.1 architecture.
Tokenizer Loading Failed: Verify that the tokenizer directories specified in the load_clip function exist and contain the necessary tokenizer files.
Out of Memory Errors: Generating high-resolution images or using a large number of inference steps can consume a significant amount of GPU memory. Try reducing the image size, the number of inference steps, or using a GPU with more memory.
Test Image Quality Issues: If the test image quality is poor, it might indicate issues with the model files or the generation parameters. Experiment with different prompts and settings.
Disclaimer
This script is provided as-is without any warranty. The user is responsible for ensuring the correct setup and usage of the script and the associated model files.

Credits
This script utilizes components from the Diffusers and Transformers libraries by Hugging Face, as well as the gguf library for loading GGUF models. The model architectures and weights are the work of their respective creators.