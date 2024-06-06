# Perform verbalization
import os
from openai import OpenAI
from utils.utils import encode_image



# Function to verbalize the image

class OpenAIVerbalizer:
    def __init__(self, model_name="gpt-4o", max_tokens=200):
        self.openai_client = OpenAI(
            api_key=os.getenv("openaikey"))
        assert(model_name in [model.id for model in self.openai_client.models.list().data])
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def verbalize_image(self, image_path):
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Explain what is in the image."},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                            }
                            }
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            print(e)
            return "Image here"
        return response.choices[0].message.content
    
    def __call__(self, image_path):
        self.verbalize_image(image_path)