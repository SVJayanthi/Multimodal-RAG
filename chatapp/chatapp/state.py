import os
import reflex as rx
from openai import OpenAI
import asyncio
import requests
import json
from typing import List, Tuple


# Checking if the API key is set properly
if not os.getenv("openaikey"):
    raise Exception("Please set openaikey environment variable.")


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str
    sources: List[Tuple[int, str]]


DEFAULT_CHATS = {
    "My Chat": [],
}


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS
    
    # The current chat name.
    current_chat = "My Chat"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""
    
    backend_url: str = "http://localhost:8001/answer"
    
    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        model = self.call_backend

        async for value in model(question):
            yield value
            
    
    async def call_backend(self, question: str):
        # Add the question to the list of questions.
        qa = QA(question=question, answer="", sources=[])
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield
        
        url = self.backend_url

        data = {"message": question}
        json_data = json.dumps(data)

        response = requests.get(url, params={"message_json": json_data})

        answer_text = "Could not complete request..."
        images = []
        if response.status_code == 200:
            resp_json = response.json()
            answer_text = resp_json['result']
            images = resp_json['source_images'].split(";")
        else:
            print("Error:", response.text)
        
        self.chats[self.current_chat][-1].sources = [(idx+1, image_loc) for idx, image_loc in enumerate(images)]
        
        # # Ensure answer_text is not None before concatenation
        if answer_text is not None:
            for i in range(len(answer_text)):
                # Pause to show the streaming effect.
                await asyncio.sleep(0.02)
                self.chats[self.current_chat][-1].answer += answer_text[i]
                
                yield
        else:
            # Handle the case where answer_text is None, perhaps log it or assign a default value
            # For example, assigning an empty string if answer_text is None
            answer_text = ""
            self.chats[self.current_chat][-1].answer += answer_text                
        self.processing = False
