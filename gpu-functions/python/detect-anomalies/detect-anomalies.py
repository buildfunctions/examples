"""
detect-anomalies

Python GPU Function for anomaly detection using PyTorch and Transformers. Built and deployed on Buildfunctions.

```requirements (add these within the Buildfunctions dashboard)
transformers
accelerate
pillow
```
"""

import json
import requests
import torch
import time
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Global variables for caching
model = None
processor = None
device = None

def initialize_model():
    """
    Initialize the model and processor if they are not already loaded,
    and set the device to GPU if CUDA is available; otherwise, default to CPU.
    """
    global model, processor, device

    if model is not None and processor is not None:
        print("Model and processor already initialized.")
        return

    start_time = time.time()

    # Determine the device (Use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

    # Model path
    model_path = "/mnt/storage/Llama-3.2-11B-Vision-Instruct-bnb-4bit"

    try:
        print("Loading model configuration...")
        config = AutoConfig.from_pretrained(model_path)

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("Loading processor...")
        processor_local = AutoProcessor.from_pretrained(model_path)

        print("Creating quantization config...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        print("Loading model weights...")
        model_local = AutoModelForImageTextToText.from_pretrained(
            model_path,
            config=config,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model_local.to(device)

        # Tie weights to resolve potential warnings
        model_local.tie_weights()

        # Assign to global variables
        model = model_local
        processor = processor_local

    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

    end_time = time.time()
    print(f"Model initialization completed in {end_time - start_time:.2f} seconds.")


def handler(event=None, context=None):
    """
    Handler function for detecting anomalies in a manufacturing process.
    Logs time taken.
    """
    initialize_model()

    timings = {}
    start_time = time.time()

    # Image URL for analysis
    image_url = (
        "https://www.dropbox.com/scl/fi/2k1u4z6tebq72ilfboj53/anomaly-test.png?rlkey=9h264rnjryp3roxv3f6iybk9o&st=wt3qupam&raw=1"
    )

    # Download and preprocess the image
    try:
        print("Downloading the image...")
        image_start = time.time()
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        timings["image_download"] = time.time() - image_start
    except Exception as e:
        print(f"Error loading the image: {e}")
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Failed to load the image. Ensure the URL is valid."}),
        }

    # Define the prompt for anomaly detection
    prompt = (
        "You are analyzing an assembly line manufacturing process. "
        "Look at all the parts in this image and identify whether any part is broken or defective. "
        "If you find a defect, respond with: 'Defect part detected: [description of defective part].' "
        "If there are no defects, respond with 'No anomalies found.'"
    )

    # Prepare the model input
    input_text = f"<|image|> {prompt}"
    preprocess_start = time.time()
    inputs = processor(images=image, text=input_text, return_tensors="pt").to(device)
    timings["preprocess"] = time.time() - preprocess_start

    # Perform inference
    try:
        print("Analyzing the image for anomalies...")
        inference_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=125,
                min_length=50,
                early_stopping=True,
                num_beams=1
            )
        timings["inference"] = time.time() - inference_start

        # Decode the raw result
        decode_start = time.time()
        raw_result = processor.decode(outputs[0], skip_special_tokens=True).strip()
        timings["decode"] = time.time() - decode_start

        # Remove the question/prompt if echoed
        result_without_prompt = raw_result.replace(prompt, "").strip()

        # Extract only the first three sentences
        sentences = result_without_prompt.split(".")  # Split into sentences
        result = ". ".join(sentences[:3]).strip() + "."  # Keep only the first three sentences

    except Exception as e:
        print(f"Error during inference: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Error during model inference. Check logs for details."}),
        }

    # Log total time taken
    timings["total_time"] = time.time() - start_time
    
    # Log the timings to the console
    print("Timings for each step:")
    for step, duration in timings.items():
        print(f"{step}: {duration:.2f} seconds")

    # Construct the final textual response
    response_body = (
        f"Prompt:\n\n{prompt}\n\n"
        f"Output:\n\n{result}\n\n"
        f"Times:\n\n{json.dumps(timings, indent=4)}\n"
    )

    # Return as plain text for easy rendering
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/plain"},
        "body": response_body
    }
