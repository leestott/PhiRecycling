from PIL import Image  # Importing the Python Imaging Library to handle image operations
import requests  # Importing requests to handle HTTP requests
import torch  # Importing PyTorch for tensor operations and model handling
from transformers import AutoModelForCausalLM  # Importing the AutoModelForCausalLM class from transformers to load the language model
from transformers import AutoProcessor  # Importing the AutoProcessor class from transformers to handle preprocessing

# Using Downloaded Model from Hugging Face
# Ensure you update this location with the model downloaded from Hugging Face
model_id = "C:\\Users\\leestott\\.cache\\huggingface\\hub\\models--microsoft--Phi-3.5-vision-instruct\\snapshots\\4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca"  # Define the model ID for the pretrained model if you want to change the model simply replace microsoft/Phi-3-vision-128k-instruct

# Define a dictionary to hold keyword arguments for model loading
# kwargs = {}
# kwargs['torch_dtype'] = torch.bfloat16  # Set the torch data type to bfloat16 for reduced precision

# Load the processor and model using the specified model ID
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 


# Function to handle online images
def call_with_single_online_image(image_url, input_prompt):
    # Construct the prompt with user input and predefined templates
    messages = [ 
        {"role": "user", "content": "<|image_1|>\n{input_prompt}"}
    ] 
    # Open the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process the prompt and image, and move the tensors to GPU
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 
    
    # Generate a response from the model
    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 0.3, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
    
    # Slice the generated IDs to remove the input prompt tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]


    # Decode the generated token IDs to get the response text
    response = processor.batch_decode(generate_ids, 
                                      skip_special_tokens=True,  # Skip special tokens in the output
                                      clean_up_tokenization_spaces=False)[0]  # Do not clean up tokenization spaces
    return response  # Return the response text

# Function to handle local images
def call_with_single_local_image(imagePath, input_prompt):
    # Construct the prompt with user input and predefined templates
    messages = [ 
        {"role": "user", "content": "<|image_1|>\n{input_prompt}"}
    ]

    # Open the image from the provided local path
    image = Image.open(imagePath, 'r')

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process the prompt and image, and move the tensors to GPU
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    
    # Generate a response from the model
    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 0.3, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
    
    # Generate a sequence of token IDs from the model using the provided inputs
    # - max_new_tokens: The maximum number of new tokens to generate
    # - eos_token_id: The token ID that represents the end of the sequence
    generate_ids = model.generate(**inputs, 
                                  max_new_tokens=100,
                                  eos_token_id=processor.tokenizer.eos_token_id)
    
    # Slice the generated IDs to remove the input prompt tokens
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

    # Decode the generated token IDs to get the response text
    # - skip_special_tokens: Skip special tokens in the output
    # - clean_up_tokenization_spaces: Do not clean up tokenization spaces


    # Decode the generated token IDs to get the response text
    response = processor.batch_decode(generate_ids, 
                                      skip_special_tokens=True,  # Skip special tokens in the output
                                      clean_up_tokenization_spaces=False)[0]  # Do not clean up tokenization spaces
    
    return response  # Return the response text
