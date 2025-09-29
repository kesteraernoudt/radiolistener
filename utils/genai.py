import google.genai as genai
from utils import logger
import os

class GenAIHandler:
    MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, pre_prompt_file: str = ""):
        self.client = genai.Client(api_key=api_key)
        self.pre_prompt_file = pre_prompt_file
        self._pre_prompt_mtime = None
        self.PRE_PROMPT = ""
        self._load_pre_prompt()

    def _load_pre_prompt(self):
        if self.pre_prompt_file and os.path.exists(self.pre_prompt_file):
            self._pre_prompt_mtime = os.path.getmtime(self.pre_prompt_file)
            with open(self.pre_prompt_file, "r") as file:
                self.PRE_PROMPT = file.read()
            logger.log_ai_event(f"GenAIHandler loaded pre_prompt: {self.PRE_PROMPT}")

    def _check_pre_prompt_update(self):
        if self.pre_prompt_file and os.path.exists(self.pre_prompt_file):
            mtime = os.path.getmtime(self.pre_prompt_file)
            if mtime != self._pre_prompt_mtime:
                self._load_pre_prompt()

    def generate(self, prompt: str, max_output_tokens: int = 1024) -> str:
        self._check_pre_prompt_update()
        logger.log_ai_event(prompt)
        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=self.PRE_PROMPT + prompt
            )
            logger.log_ai_event(f"GenAIHandler response: {response.text}")
            return response.text
        except Exception as e:
            print(f"GenAIHandler generate error: {e}")
            logger.log_ai_event(f"GenAIHandler generate error: {e}")
            return ""
        
if __name__ == "__main__":
    genai_handler = GenAIHandler(api_key="MY_API_KEY")
    test_prompt = "Detects that to us right now. You couldn't win a family four pack to six flags great America. Grizzly 408 516 1065 We got all Navy get 50 percent"

    print(genai_handler.generate(test_prompt))

    test_prompt2 = "long. All you need to do to qualify is text this word to us right now. What is demon as a demon drop? D-E-M-O-N. That's a spell. Demon. Text that to us right now.  Hexat to us right now and you and the family could be checking out six flags great America That's a 408 506 1065 looking for the top sales at Rayleigh's and I'm"
    print(genai_handler.generate(test_prompt2))

