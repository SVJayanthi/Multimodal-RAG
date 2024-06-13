# Perform verbalization
import os
from openai import OpenAI
from utils.utils import encode_image

# Function to verbalize the image

class OpenAILMM:
    def __init__(self, model_name="gpt-4o", max_tokens=200):
        self.openai_client = OpenAI(
            api_key=os.getenv("openaikey"))
        assert(model_name in [model.id for model in self.openai_client.models.list().data])
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def call_lmm(self, prompt, images):
        # image_list = [
        #                     {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/jpeg;base64,{encode_image(i)}"
        #                     }
        #                     } for i in images]
        text_list = [{"type": "text", "text": prompt}]
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": text_list,
                    }
                ],
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            print(e)
            return "LMM Failed, please try again"
        return response.choices[0].message.content
    
    def __call__(self, prompt, images):
        return  self.call_lmm(prompt, images)