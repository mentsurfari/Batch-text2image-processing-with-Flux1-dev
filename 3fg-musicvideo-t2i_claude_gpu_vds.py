import csv
import time
import os
import torch
import gguf
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer, CLIPTextConfig, T5Config
import safetensors.torch
from PIL import Image  # Added PIL import at the top level
import sys
import gc
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

def map_keys(gguf_key):
    """
    Maps GGUF tensor names to Diffusers Flux UNET state dict keys.
    Based on observed GGUF names (e.g., double_blocks.0.img_attn.*).
    """
    if gguf_key.startswith("double_blocks"):
        gguf_key = gguf_key.replace("double_blocks", "transformer_blocks")
    if "img_attn" in gguf_key:
        gguf_key = gguf_key.replace("img_attn", "attn")
        if ".qkv." in gguf_key:
            gguf_key = gguf_key.replace(".qkv.", ".to_qkv.")
        elif ".proj." in gguf_key:
            gguf_key = gguf_key.replace(".proj.", ".to_out.")
        elif ".norm." in gguf_key:
            gguf_key = gguf_key.replace(".norm.", ".norm1.")
    if gguf_key.endswith(".scale"):
        gguf_key = gguf_key.replace(".scale", ".weight")
    return gguf_key

def load_gguf_unet(model_path, device):
    """Load UNET weights from GGUF file with a hardcoded config."""
    print("Loading GGUF model (this may take a while)...")
    gguf_model = gguf.GGUFReader(model_path)
    print("Extracting tensors...")
    tensors = {tensor.name: torch.tensor(tensor.data) for tensor in gguf_model.tensors}
    print("Mapping keys...")
    state_dict = {map_keys(k): v.to(dtype=torch.float16) for k, v in tensors.items()}
    
    # Free memory
    del gguf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hardcode FLUX.1 UNET configuration (based on typical FLUX.1 architecture)
    unet_config = {
        "sample_size": 64,
        "in_channels": 16,
        "out_channels": 16,
        "layers_per_block": 2,
        "attention_head_dim": [8, 16, 32, 64],
        "cross_attention_dim": [768, 768, 768, 768],
        "block_out_channels": [320, 640, 1280, 2560],
        "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
        "transformer_layers_per_block": 1,
        "use_linear_projection": True,
        "class_embed_type": None,
        "addition_embed_type": None,
        "projection_class_embeddings_input_dim": None,
        "dual_cross_attention": False,
    }

    # Initialize UNET
    print("Initializing UNet...")
    unet = UNet2DConditionModel(**unet_config)
    unet = unet.to(dtype=torch.float16)
    print(f"Moving UNet to {device}...")
    unet.to(device)
    
    print("Loading UNet weights...")
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    print(f"UNET loading: Missing keys: {len(missing)} (Sample: {missing[:5]})")
    print(f"UNET loading: Unexpected keys: {len(unexpected)} (Sample: {unexpected[:5]})")
    
    # Free memory
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return unet

def load_vae(vae_path, device):
    print("Loading VAE weights...")
    state_dict = safetensors.torch.load_file(vae_path)
    
    # Corrected VAE config for FLUX.1 model
    # The key fix is making sure latent_channels=16 and other params match exactly
    vae_config = {
        "sample_size": 512,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 16,  # Critical for FLUX models
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2,
        "act_fn": "silu",
        "scaling_factor": 0.18215,  # Standard scaling factor for most VAEs
    }
    
    # Initialize VAE with updated config
    print("Initializing VAE with updated config...")
    vae = AutoencoderKL(**vae_config)
    vae = vae.to(dtype=torch.float16)
    print(f"Moving VAE to {device}...")
    vae.to(device)
    
    # Load weights with strict=False
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print(f"VAE Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")
    
    # Free memory
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return vae

def load_clip(clip_path, t5_path, device):
    """Load CLIP and T5 encoder from local files with corrected config."""
    print("Loading CLIP weights...")
    clip_state_dict = safetensors.torch.load_file(clip_path)
    
    print("Loading T5 weights (this may take a while)...")
    t5_model = gguf.GGUFReader(t5_path)
    t5_state_dict = {t.name: torch.tensor(t.data) for t in t5_model.tensors}
    
    # Free memory
    del t5_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Corrected CLIP configuration
    clip_config = CLIPTextConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        projection_dim=768,
    )
    
    # Corrected T5 configuration
    t5_config = T5Config(
        d_model=2048,
        num_layers=24,
        num_heads=32,
        d_ff=5120,
        dropout_rate=0.1,
    )
    
    print("Initializing models...")
    clip = CLIPTextModel(clip_config)
    t5 = T5EncoderModel(t5_config)
    
    # Cast to float16 for consistency and move to device
    clip = clip.to(dtype=torch.float16)
    t5 = t5.to(dtype=torch.float16)
    
    print(f"Moving CLIP and T5 to {device}...")
    clip.to(device)
    t5.to(device)
    
    # Load weights
    print("Loading model weights...")
    clip.load_state_dict(clip_state_dict, strict=False)
    t5.load_state_dict(t5_state_dict, strict=False)
    
    # Free memory
    del clip_state_dict, t5_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load tokenizers
    clip_tokenizer_dir = "C:/Users/userr/Documents/ComfyUI/models/clip_tokenizer"
    t5_tokenizer_dir = "C:/Users/userr/Documents/ComfyUI/models/t5_tokenizer"

    # Load CLIP tokenizer with validation
    print("Loading tokenizers...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            clip_tokenizer_dir, 
            local_files_only=True
        )
        if not tokenizer:
            raise ValueError("CLIP tokenizer initialization failed")
        print("CLIP tokens:", tokenizer("Test prompt").input_ids)
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP tokenizer: {e}")
    
    # Load T5 tokenizer with validation
    try:
        t5_tokenizer = T5Tokenizer.from_pretrained(
            t5_tokenizer_dir, 
            local_files_only=True,
            legacy=True
        )
        if not t5_tokenizer:
            raise ValueError("T5 tokenizer initialization failed")
        print("T5 tokens:", t5_tokenizer("Test prompt").input_ids)
    except Exception as e:
        raise RuntimeError(f"Failed to load T5 tokenizer: {e}")
    
    return clip, t5, tokenizer, t5_tokenizer

def update_progress(current, total, description=""):
    """Display a progress bar in the console."""
    bar_length = 40
    filled_length = int(round(bar_length * current / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r{description} |{bar}| {current}/{total} [{current/total:.0%}]')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')

def get_timesteps(scheduler_name="euler", num_inference_steps=20):
    """
    Get properly spaced timesteps for the denoising process.
    Matches the ComfyUI scheduler implementation more closely.
    """
    # Initial timestep sequence based on linear spacing
    if scheduler_name == "euler":
        # FLUX models use 0-999 timestep range
        timesteps = torch.linspace(999, 0, num_inference_steps)
        return timesteps.long()
    else:
        # Default to linear steps
        return torch.linspace(999, 0, num_inference_steps).long()

def create_empty_latent(batch_size=1, height=512, width=512, device="cuda"):
    """Create an empty latent image with FLUX model compatible dimensions."""
    # For FLUX models, the latent space uses 16 channels and downsampling factor of 8
    channels = 16  # Critical for FLUX models
    latent_height = height // 8
    latent_width = width // 8
    
    # Create random latent with proper distribution
    latent = torch.randn(
        (batch_size, channels, latent_height, latent_width),
        device=device,
        dtype=torch.float16
    )
    
    return latent

def encode_prompt(prompt, clip, t5, clip_tokenizer, t5_tokenizer, device):
    """Encode the text prompt using both CLIP and T5 encoders."""
    # Encode with CLIP
    clip_inputs = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Encode with T5
    t5_inputs = t5_tokenizer(
        prompt,
        padding="max_length", 
        max_length=256,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    print("Generating embeddings...")
    with torch.no_grad():
        # Get CLIP embeddings
        clip_embeds = clip(
            input_ids=clip_inputs.input_ids,
            attention_mask=clip_inputs.attention_mask
        ).last_hidden_state
        
        # Get T5 embeddings
        t5_embeds = t5(
            input_ids=t5_inputs.input_ids,
            attention_mask=t5_inputs.attention_mask
        ).last_hidden_state
        
        print(f"CLIP embedding shape: {clip_embeds.shape}")
        print(f"T5 embedding shape: {t5_embeds.shape}")
    
    # For FLUX.1, we'll use the CLIP embeddings
    return clip_embeds

def create_negative_prompt_embeds(clip, clip_tokenizer, device):
    """Create negative prompt embeddings with an empty string."""
    empty_prompt = ""
    negative_inputs = clip_tokenizer(
        empty_prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        negative_embeds = clip(
            input_ids=negative_inputs.input_ids,
            attention_mask=negative_inputs.attention_mask
        ).last_hidden_state
    
    return negative_embeds

def euler_sampling_step(model, latents, timestep, guidance_scale, encoder_hidden_states, negative_encoder_hidden_states):
    """Perform a single Euler sampling step with classifier-free guidance."""
    # Expand the latents if doing classifier-free guidance
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
    
    # Get timestep representation for the model
    t_input = timestep.expand(latent_model_input.shape[0])
    
    # Combine positive and negative conditioning
    hidden_states = torch.cat([negative_encoder_hidden_states, encoder_hidden_states]) if guidance_scale > 1.0 else encoder_hidden_states
    
    # Get the noise prediction
    with torch.no_grad():
        noise_pred = model(
            latent_model_input,
            t_input,
            encoder_hidden_states=hidden_states
        ).sample
    
    # Perform guidance
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # Compute step size for Euler step
    step_size = 1000 // 20  # For t from 999 to 0 with 20 steps
    
    # Euler step (simplified for FLUX models)
    step_output = latents - step_size * noise_pred
    
    return step_output

def decode_latents_with_vae(vae, latents):
    """
    Decode latents to image using VAE decoder - completely rewritten
    for FLUX model compatibility.
    """
    # Ensure latents are on the same device as VAE
    latents = latents.to(vae.device)
    
    # Apply correct scaling factor for FLUX models
    # This is critical - the scaling factor ensures proper normalization
    scaled_latents = latents / vae.config.scaling_factor
    
    # Decode the latents
    with torch.no_grad():
        # Use the proper VAE decoder method
        decoded = vae.decode(scaled_latents).sample
        
        # Process the output images
        images = decoded
        
        # Normalize to [0, 1] range for PIL
        images = (images / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy array
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Convert to uint8 for PIL
        images = (images * 255).round().astype("uint8")
        
        return images[0]  # Return first image (batch size 1)

def improved_emergency_decode(latents, height, width):
    """
    Significantly improved emergency decoding function.
    Uses advanced normalization techniques to get a more visually
    coherent result from the latent space.
    """
    print("Using improved emergency decoding...")
    
    # Get latents to CPU and float32
    latents_np = latents.detach().cpu().numpy()[0]  # Remove batch dimension
    
    # FLUX models use 16 channels, but we'll extract a good RGB interpretation
    # Use statistical properties to get the most informative channels
    
    # Calculate variance of each channel
    channel_variance = np.var(latents_np, axis=(1, 2))
    
    # Get indices of channels with highest variance (most information)
    best_channels = np.argsort(channel_variance)[-3:]
    
    # Create an RGB image from the most informative channels
    rgb_image = np.zeros((latents_np.shape[1], latents_np.shape[2], 3))
    
    # Map the best channels to RGB
    for i, channel_idx in enumerate(best_channels):
        channel = latents_np[channel_idx]
        # Normalize each channel independently with robust normalization
        p_low, p_high = np.percentile(channel, [2, 98])
        rgb_image[:, :, i] = np.clip((channel - p_low) / (p_high - p_low + 1e-5), 0, 1)
    
    # Resize to target dimensions using high-quality interpolation
    from skimage.transform import resize
    rgb_image = resize(rgb_image, (height, width, 3), order=3, anti_aliasing=True)
    
    # Convert to uint8
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return rgb_image

def generate_image(prompt, unet, vae, clip, t5, clip_tokenizer, t5_tokenizer, device, timeout=300, 
                   guidance_scale=7.5, num_inference_steps=30, width=512, height=512, seed=None):
    """Generate an image using the loaded models with proper sampling."""
    if not prompt or not isinstance(prompt, str):
        raise ValueError(f"Invalid prompt: {prompt}")
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        # Use a random seed
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    print(f"Using seed: {seed}")
    
    # Create initial latent noise
    latents = create_empty_latent(batch_size=1, height=height, width=width, device=device)
    
    # Encode prompt
    encoder_hidden_states = encode_prompt(prompt, clip, t5, clip_tokenizer, t5_tokenizer, device)
    
    # Create negative prompt embeddings
    negative_encoder_hidden_states = create_negative_prompt_embeds(clip, clip_tokenizer, device)
    
    # Get timesteps for sampling - fixed for FLUX models
    timesteps = get_timesteps(scheduler_name="euler", num_inference_steps=num_inference_steps)
    timesteps = timesteps.to(device)
    
    print("Starting denoising process...")
    start_time = time.time()
    time_limit = start_time + timeout
    
    # Perform denoising steps
    for i, t in enumerate(timesteps):
        current_time = time.time()
        if current_time > time_limit:
            raise TimeoutError(f"Image generation timed out after {timeout} seconds")
        
        update_progress(i+1, len(timesteps), f"Denoising step {i+1}/{len(timesteps)}")
        
        # Perform a single denoising step
        latents = euler_sampling_step(
            unet, 
            latents, 
            t, 
            guidance_scale, 
            encoder_hidden_states, 
            negative_encoder_hidden_states
        )
        
        # Add small periodic noise to break repetitive patterns (helps with some artifacts)
        if i > num_inference_steps // 2:  # Only in latter half of generation
            noise_level = 0.002 * (1.0 - i / len(timesteps))  # Decreases as we approach the end
            latents = latents + noise_level * torch.randn_like(latents)
    
    print("\nDecoding image with VAE...")
    
    # First try the VAE decoder - the proper way
    try:
        image_array = decode_latents_with_vae(vae, latents)
        print("VAE decoding successful!")
    except Exception as e:
        print(f"VAE decoding failed with error: {e}")
        print("Trying improved emergency decoding method...")
        
        try:
            # Use the improved emergency decoding
            image_array = improved_emergency_decode(latents, height, width)
            print("Emergency decoding successful")
        except Exception as e2:
            print(f"Emergency decoding also failed: {e2}")
            
            # Last resort fallback to simple visualization
            print("Creating error placeholder image...")
            # Create a placeholder image - don't rely on local Image import in the function
            placeholder = Image.new('RGB', (width, height), color=(30, 30, 30))
            draw = ImageDraw.Draw(placeholder)
            draw.text((width//10, height//2), f"Decoding error: {str(e)[:50]}...", fill=(255, 50, 50))
            return placeholder
    
    # Clean up to free memory
    del latents, encoder_hidden_states, negative_encoder_hidden_states
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create a PIL Image from the array
    print("Converting to PIL Image...")
    try:
        # Use the PIL module imported at the top level
        image = Image.fromarray(image_array)
        print("Image generation complete")
        return image
    except Exception as e:
        print(f"Failed to create PIL image: {e}")
        # Add more detailed error information
        print(f"Image array shape: {image_array.shape if hasattr(image_array, 'shape') else 'unknown'}")
        print(f"Image array dtype: {image_array.dtype if hasattr(image_array, 'dtype') else 'unknown'}")
        print(f"Image array min/max: {np.min(image_array) if hasattr(image_array, 'min') else 'unknown'} / "
              f"{np.max(image_array) if hasattr(image_array, 'max') else 'unknown'}")
        raise

def check_gpu():
    """Enhanced GPU check with detailed diagnostics."""
    if not torch.cuda.is_available():
        print("\nERROR: CUDA is not available. Cannot use GPU.")
        print("Possible solutions:")
        print("1. Install CUDA-compatible PyTorch: https://pytorch.org/get-started/locally/")
        print("2. Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
        print("3. Verify GPU compatibility: https://developer.nvidia.com/cuda-gpus")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"\nFound {device_count} CUDA device(s):")
    
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    
    try:
        device = torch.device("cuda:0")
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        _ = test_tensor * 2  # Simple operation test
        print(f"CUDA functionality verified on {device}")
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"CUDA functionality test failed: {e}")
        print("This may indicate:")
        print("- Insufficient GPU memory")
        print("- Driver/CUDA version mismatch")
        print("- Hardware compatibility issues")
        return False

def main():
    try:
        gpu_available = check_gpu()
        if not gpu_available:
            print("\nWARNING: GPU acceleration unavailable. CPU mode will be very slow.")
            if input("Continue with CPU? (y/n): ").lower() != 'y':
                print("Exiting program.")
                return
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        
        print(f"\nUsing device: {device}")
        
        # Define file paths
        unet_path = "C:/Users/userr/Documents/ComfyUI/models/diffusion_models/FLUX1/flux1-dev-Q4_K_S.gguf"
        vae_path = "C:/Users/userr/Documents/ComfyUI/models/vae/FLUX1/ae.safetensors"
        clip_path = "C:/Users/userr/Documents/ComfyUI/models/text_encoders/clip_l.safetensors"
        t5_path = "C:/Users/userr/Documents/ComfyUI/models/text_encoders/t5/t5-v1_1-xxl-encoder-Q4_K_S.gguf"
        csv_path = "C:/Users/userr/Documents/GitHub/video-creation-automation/[hiphop]giftoftnothing_10_beat_clips/all_transcriptions.csv"
        output_folder = "C:/Users/userr/Documents/GitHub/video-creation-automation/[hiphop]giftoftnothing_10_beat_clips/generated_images"

        # Validate paths exist before proceeding
        for path, name in [(unet_path, "UNET"), (vae_path, "VAE"), (clip_path, "CLIP"), (t5_path, "T5"), (csv_path, "CSV")]:
            if not os.path.exists(path):
                print(f"ERROR: {name} file not found at: {path}")
                return

        os.makedirs(output_folder, exist_ok=True)

        print("Loading UNET from GGUF...")
        unet = load_gguf_unet(unet_path, device)
        
        print("Loading VAE...")
        vae = load_vae(vae_path, device)
        
        print("Loading CLIP and T5...")
        clip, t5, clip_tokenizer, t5_tokenizer = load_clip(clip_path, t5_path, device)
        
        # Test with a small image first to verify pipeline
        try:
            test_size = 256  # Smaller test size to save memory
            test_prompt = "A beautiful landscape with mountains and a lake, detailed, high quality"
            print(f"\nTesting with prompt: '{test_prompt}'")
            # Use fixed seed for reproducibility in testing
            test_image = generate_image(
                test_prompt, 
                unet, 
                vae, 
                clip, 
                t5, 
                clip_tokenizer, 
                t5_tokenizer, 
                device,
                seed=42,
                num_inference_steps=15,  # Reduced for test
                width=test_size,
                height=test_size,
                guidance_scale=7.5
            )
            test_path = os.path.join(output_folder, "test_image.png")
            test_image.save(test_path)
            print(f"Test image saved to {test_path}")
            
            # Ask user to verify test image quality
            print("\nIMPORTANT: Please check the test image quality at:")
            print(test_path)
            proceed = input("Is the test image quality acceptable? (y/n): ").lower()
            if proceed != 'y':
                print("Test image quality not acceptable. Please check the model files and try again.")
                return
                
        except Exception as e:
            print(f"Test image generation failed: {e}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            if input("\nContinue with CSV processing despite test failure? (y/n): ").lower() != 'y':
                print("Exiting program.")
                return

        print("Reading CSV file...")
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Store header
                print(f"CSV header: {header}")
                
                # Check if the CSV has at least 2 columns
                if len(header) < 2:
                    print(f"WARNING: CSV has only {len(header)} columns, expecting at least 2.")
                    column_index = 0  # Default to first column
                else:
                    column_index = 1  # Use second column by default
                
                prompts = []
                for row in reader:
                    if not row:  # Skip empty rows
                        continue
                    if len(row) <= column_index:
                        print(f"WARNING: Row {len(prompts)+1} has insufficient columns: {row}")
                        continue
                    
                    prompt = row[column_index].strip()
                    if not prompt:  # Skip empty prompts
                        continue
                    prompts.append(prompt)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            import traceback
            traceback.print_exc()
            return

        if not prompts:
            print("No valid text entries found in the CSV file.")
            return

        total_prompts = len(prompts)
        print(f"Found {total_prompts} valid prompts in CSV file.")
         
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{total_prompts}] Generating image for prompt: '{prompt}'")
            start_time = time.time()

            try:
                # Generate a unique seed for each image
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                
                # Enhanced generation parameters
                image = generate_image(
                    prompt, 
                    unet, 
                    vae, 
                    clip, 
                    t5, 
                    clip_tokenizer, 
                    t5_tokenizer, 
                    device,
                    seed=seed,
                    num_inference_steps=40,  # Increased for better quality
                    guidance_scale=8.0,      
                    width=512,
                    height=512
                )
            

                image_file = f"image_{i}.png"
                image_path = os.path.join(output_folder, image_file)
                image.save(image_path)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Image saved to {image_path} (Time taken: {elapsed_time:.2f} seconds)")
                
                # Clear GPU memory after each image generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                print("Continuing with next prompt...")
                continue
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return        


if __name__ == "__main__":
    main()