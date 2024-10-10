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
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "Phi-3.5-vision-instruct" # Example to change this to from Phi-3.5 SLM "Phi-3.5-vision-instruct" to a LLM such as GPT4o replace with "gpt-4o"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


def call_with_single_local_image(imagePath, input_prompt):

    response = client.complete(
        messages=[
            UserMessage(
                content=[
                    TextContentItem(text=input_prompt),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file=imagePath,
                            image_format="jpg",
                            detail=ImageDetailLevel.AUTO)
                    ),
                ],
            ),
        ],
        model=model_name,
    )

    return response.choices[0].message.content