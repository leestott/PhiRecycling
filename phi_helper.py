# Import the libraries, Importing the os module to access environment variables 
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
)
from azure.core.credentials import AzureKeyCredential

# Retrieve the GitHub token from environment variables
token = os.environ["GITHUB_TOKEN"]

# Define the endpoint for the Azure AI inference service
endpoint = "https://models.inference.ai.azure.com"

# Define the model name to be used for inference
model_name = "Phi-3.5-vision-instruct" # Example to change this to from Phi-3.5 SLM "Phi-3.5-vision-instruct" to a LLM such as GPT4o replace with "gpt-4o"

# Initialize the ChatCompletionsClient with the endpoint and credentials
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# Function to call the model with a single local image and input prompt
def call_with_single_local_image(imagePath, input_prompt):
    
    # Create a response by sending a completion request to the client
    response = client.complete(
        messages=[
            UserMessage(
                content=[
                    TextContentItem(text=input_prompt),  # Add the input prompt as text content
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file=imagePath, # Load the image from the provided local path
                            image_format="jpg",  # Specify the image format
                            detail=ImageDetailLevel.AUTO) # Set the detail level to auto
                    ),
                ],
            ),
        ],
        model=model_name, # Specify the model name for inference
    )

    # Return the content of the first choice in the response
    return response.choices[0].message.content