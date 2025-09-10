import google.genai as genai

class GenAIHandler:
    MODEL = "gemini-2.5-flash"
    PRE_PROMPT = "The following is a transcript from a radio show where there might be a code word that needs to be texted. If you find that word, return just that codeword (usually one word, but might be more). If you don't find such code word, return an empty string. Don't give any reasoning either. Here is the transcript: "

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, max_output_tokens: int = 1024) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=self.PRE_PROMPT + prompt
            )
            return response.text
        except Exception as e:
            print(f"GenAIHandler generate error: {e}")
            return ""
        
if __name__ == "__main__":
    genai_handler = GenAIHandler(api_key="MY_API_KEY")
    test_prompt = "Detects that to us right now. You couldn't win a family four pack to six flags great America. Grizzly 408 516 1065 We got all Navy get 50 percent"

    print(genai_handler.generate(test_prompt))

    test_prompt2 = "long. All you need to do to qualify is text this word to us right now. What is demon as a demon drop? D-E-M-O-N. That's a spell. Demon. Text that to us right now.  Hexat to us right now and you and the family could be checking out six flags great America That's a 408 506 1065 looking for the top sales at Rayleigh's and I'm"
    print(genai_handler.generate(test_prompt2))

