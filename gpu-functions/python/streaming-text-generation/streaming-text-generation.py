"""
Name: streaming-text-generation

Python GPU Function for text generation with a streaming response using PyTorch and Transformers. Built and deployed on Buildfunctions.

```requirements (add these within the Buildfunctions dashboard)
transformers
accelerate
```
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Global variables for caching the model, tokenizer, and device.
model = None
tokenizer = None
device = None

def initialize_model():
    """
    This function initializes the model, tokenizer, and device by loading them into memory and automatically selecting the GPU if available. Caching is used to ensure the initialization runs only once.
    """
    global model, tokenizer, device

    try:
        # If the model and tokenizer are already initialized, do nothing.
        if model is not None and tokenizer is not None:
            return

        # Set the device to GPU if available, otherwise use CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True  # Optimize GPU performance
        
        # Path to the model (can be local or remote)
        model_path = "/mnt/storage/Llama-3.2-3B-Instruct-bnb-4bit"

        # Load the model configuration and tokenizer from the specified path.
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Enable 4-bit quantization for memory optimization.
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        # Load the model with 4-bit quantization and map it to the appropriate device.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,  # Use half-precision for faster computation.
            device_map="auto",  # Automatically map the model layers to the available devices.
            quantization_config=bnb_config if not hasattr(config, "quantization_config") else None,
        )
        model.to(device)  # Move the model to the chosen device.

    except Exception as e:
        print(f"Error initializing the model: {e}")
        raise RuntimeError(
            "Failed to initialize the model. Please check the model path and configuration."
        )


async def stream_tokens(prompt):
    """
    This function streams tokens from the model incrementally, providing the output in real-time
    """
    try:
        # Ensure the model is initialized before generating tokens.
        initialize_model()

        # Tokenize the input prompt and move it to the device.
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)

        # Start the stream by emitting a header.
        yield b"<<START_STREAM>>\n"

        with torch.no_grad():
            past_key_values = None  # Store past key values for caching in subsequent iterations.
            generated_ids = input_ids  # Start with the input prompt.

            sentence_count = 0  # Track the number of sentences generated.

            for _ in range(200):  # Limit the output to a maximum of 200 tokens.
                try:
                    # Generate the next token logits
                    outputs = model(
                        input_ids=generated_ids,
                        past_key_values=past_key_values,  # Use cached values to speed up generation
                        use_cache=True,  # Enable caching of key values
                    )
                    logits = outputs.logits[:, -1, :]  # Extract logits for the last token

                    # Apply temperature and top-k sampling
                    temperature = 0.7
                    logits = logits / temperature
                    top_k = 50
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.zeros_like(logits).scatter_(1, indices, values)
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)

                    # Sample the next token
                    next_token_id = torch.multinomial(probabilities, num_samples=1)

                    # Update past key values and input IDs
                    past_key_values = outputs.past_key_values
                    generated_ids = next_token_id

                    # Decode and yield the token
                    token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
                    yield f"<<STREAM_CHUNK>>{token}<<END_STREAM_CHUNK>>\n".encode()

                    # Check for stopping conditions
                    if token in {".", "!", "?"}:
                        sentence_count += 1
                    if sentence_count >= 3 or next_token_id.squeeze().item() == tokenizer.eos_token_id:
                        break

                except Exception as gen_error:
                    print(f"Error during token generation: {gen_error}")
                    yield b"<<STREAM_ERROR>>\n"
                    break

        # Emit the end of the stream to signal completion.
        yield b"<<END_STREAM>>\n"

    except Exception as e:
        print(f"Error in streaming tokens: {e}")
        yield b"<<STREAM_ERROR>>\n"


async def async_stream_wrapper(prompt):
    """
    This function wraps the token streaming process, allowing iteration over streamed chunks in an asynchronous context.
    """
    try:
        async for chunk in stream_tokens(prompt):
            yield chunk
    except Exception as e:
        print(f"Error in async stream wrapper: {e}")
        yield b"<<STREAM_ERROR>>\n"


def handler():
    """
    This function serves as the entry point, processing the input prompt and returning a streaming response
    """
    try:
        # Use a hardcoded prompt.
        prompt = (
            "Tell me about the most mysterious phenomena in the universe."
        )

        # Return the response containing the status code, headers, and streamed body.
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            },
            "body": async_stream_wrapper(prompt),  # Use the asynchronous stream wrapper as the response body.
        }

    except Exception as e:
        print(f"Error in handler: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": {"error": "Internal Server Error"},
        }

