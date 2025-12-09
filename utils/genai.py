import os

import google.genai as genai
from utils import logger

try:
    from groq import Groq
except ImportError:  # pragma: no cover - safety if dependency missing
    Groq = None


class GenAIHandler:
    GEMINI_MODEL = "gemini-2.5-flash"
    GROQ_MODEL = "llama-3.3-70b-versatile"  # primary Groq model (override via env GROQ_MODEL)
    GROQ_FALLBACKS = [
        "llama-3.1-8b-instant",
    ]

    def __init__(self, gemini_api_key: str = "", pre_prompt_file: str = "", groq_api_key: str = "", provider: str = "auto"):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.gemini_client = genai.Client(api_key=self.gemini_api_key) if self.gemini_api_key else None
        env_model = os.getenv("GROQ_MODEL", "").strip()
        self.groq_model = env_model or self.GROQ_MODEL
        self._groq_fallbacks = list(self.GROQ_FALLBACKS)
        self.groq_client_error = ""
        if self.groq_api_key:
            if Groq is None:
                self.groq_client = None
                self.groq_client_error = "groq SDK missing; install/upgrade via `pip install -U groq`"
            else:
                try:
                    self.groq_client = Groq(api_key=self.groq_api_key)
                except Exception as exc:  # pragma: no cover - defensive
                    self.groq_client = None
                    self.groq_client_error = f"Failed to init Groq client: {exc}"
        else:
            self.groq_client = None
            self.groq_client_error = "No Groq API key provided."
        provider = (provider or "auto").lower()
        self.provider = provider if provider in ("auto", "gemini", "groq") else "auto"
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

    def _log_and_validate(self, text: str, radio: str = "") -> str | None:
        logger.log_ai_event(f"Response: {text}", radio)
        if not text:
            logger.log_ai_event("Empty AI response; skipping", radio)
            return None
        # this should be a keyword or codeword or so, not a long response. So filter out any long reply
        if len(text) > 30:
            logger.log_ai_event("This is a way too long response to be a codeword, so skipping", radio)
            return None
        return text

    def _is_rate_limit_error(self, error: Exception) -> bool:
        message = str(error).upper()
        return "RESOURCE_EXHAUSTED" in message or "429" in message or "RATE LIMIT" in message

    def _generate_with_gemini(self, prompt: str, radio: str, max_output_tokens: int) -> tuple[str | None, bool]:
        if not self.gemini_client:
            return None, True  # allow fallback when Gemini is not configured
        try:
            response = self.gemini_client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=self.PRE_PROMPT + prompt
            )
            text = getattr(response, "text", "") or ""
            return self._log_and_validate(text, radio), False
        except Exception as e:
            print(f"GenAIHandler generate error: {e}")
            logger.log_ai_event(f"GenAIHandler generate error: {e}", radio)
            return None, self._is_rate_limit_error(e)

    def _generate_with_groq(self, prompt: str, radio: str, max_output_tokens: int) -> str | None:
        if not self.groq_client:
            logger.log_ai_event("Groq requested but not configured", radio)
            reason = self.groq_client_error or "unknown reason"
            print(f"Groq client not configured; skipping Groq call. Reason: {reason}")
            return None
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "user", "content": self.PRE_PROMPT + prompt},
                ],
                max_tokens=min(max_output_tokens, 64),
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip() if response and response.choices else ""
            logger.log_ai_event("Using Groq fallback", radio)
            return self._log_and_validate(text, radio)
        except Exception as e:
            print(f"Groq generate error: {e}")
            logger.log_ai_event(f"Groq generate error: {e}", radio)
            err_str = str(e).lower()
            if "decommissioned" in err_str or "no longer supported" in err_str or "model_not_found" in err_str:
                if self._groq_fallbacks:
                    next_model = self._groq_fallbacks.pop(0)
                    logger.log_ai_event(f"Switching Groq model to fallback {next_model}", radio)
                    self.groq_model = next_model
                    return self._generate_with_groq(prompt, radio, max_output_tokens)
                logger.log_ai_event("No Groq fallback models left to try", radio)
            return None

    def generate(self, prompt: str, radio: str = "", max_output_tokens: int = 1024) -> str | None:
        self._check_pre_prompt_update()
        logger.log_ai_event(f"Context: {prompt}", radio)
        if self.provider == "groq":
            return self._generate_with_groq(prompt, radio, max_output_tokens)

        if self.provider == "gemini":
            text, _ = self._generate_with_gemini(prompt, radio, max_output_tokens)
            return text

        # auto: try gemini first, then groq on rate limit / missing key
        text, fallback = self._generate_with_gemini(prompt, radio, max_output_tokens)
        if text is not None:
            return text
        if fallback:
            return self._generate_with_groq(prompt, radio, max_output_tokens)
        return None
        
if __name__ == "__main__":
    genai_handler = GenAIHandler(gemini_api_key="MY_API_KEY")
    test_prompt = "Detects that to us right now. You couldn't win a family four pack to six flags great America. Grizzly 408 516 1065 We got all Navy get 50 percent"

    print(genai_handler.generate(test_prompt))

    test_prompt2 = "long. All you need to do to qualify is text this word to us right now. What is demon as a demon drop? D-E-M-O-N. That's a spell. Demon. Text that to us right now.  Hexat to us right now and you and the family could be checking out six flags great America That's a 408 506 1065 looking for the top sales at Rayleigh's and I'm"
    print(genai_handler.generate(test_prompt2))
