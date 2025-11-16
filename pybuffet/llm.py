## OpenAI
import openai, os

class OpenAIClient:
    
    def __init__(self, model="gpt-4o-mini"):
        self.__model = model
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        self.__client = openai.OpenAI()

    def run_prompt(self, prompt, model=None):
        if model is None:
            model = self.__model
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.__client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

__all__ = ['OpenAIClient']