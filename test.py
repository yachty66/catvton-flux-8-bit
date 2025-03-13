import os
import torch
import time
import subprocess

def run_inference(
    image_path,
    mask_path,
    garment_path,
    output_path,
    seed=4096,
    steps=5,
    height=768,  # default values
    width=576    # default values
):
    # Set cache directory
    os.environ['HUGGINGFACE_HUB_CACHE'] = './models'
    
    # Start timing
    image_gen_start_time = time.time()
    
    # Construct command as a list of arguments
    cmd = [
        "python",
        "tryon_inference_quantized.py",
        "--image", image_path,
        "--mask", mask_path,
        "--garment", garment_path,
        "--seed", str(seed),
        "--output_tryon", output_path,
        "--steps", str(steps)
    ]
    
    # Optionally add height and width if different from defaults
    if height != 768:
        cmd.extend(["--height", str(height)])
    if width != 576:
        cmd.extend(["--width", str(width)])
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        print(f"Stderr: {e.stderr}")
        raise
    
    # Calculate and print timing
    image_gen_end_time = time.time()
    print("Image Gen Seconds: ", round((image_gen_end_time - image_gen_start_time), 1))

if __name__ == "__main__":
    # Example usage
    run_inference(
        image_path="path/to/your/image.jpg",
        mask_path="path/to/your/mask.jpg",
        garment_path="path/to/your/garment.jpg",
        output_path="path/to/output.png",
        seed=4096,
        steps=5,
        # height=512,  # uncomment to override default
        # width=256    # uncomment to override default
    )