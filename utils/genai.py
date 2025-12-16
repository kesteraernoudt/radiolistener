import os

import google.genai as genai
from utils import logger

try:
    from groq import Groq
except ImportError:  # pragma: no cover - safety if dependency missing
    Groq = None


class GenAIHandler:
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_FALLBACKS = [ 
        "gemini-2.0-flash-lite"
    ]
    GROQ_MODEL = "openai/gpt-oss-120b"  # primary Groq model (override via env GROQ_MODEL)
    GROQ_FALLBACKS = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]

    def __init__(self, gemini_api_key: str = "", pre_prompt_file: str = "", groq_api_key: str = "", provider: str = "auto"):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.gemini_client = genai.Client(api_key=self.gemini_api_key) if self.gemini_api_key else None
        gemini_env_model = os.getenv("GEMINI_MODEL", "").strip()
        self.gemini_model = gemini_env_model or self.GEMINI_MODEL
        gemini_fallback_env = os.getenv("GEMINI_FALLBACKS", "")
        configured_gemini_fallbacks = (
            [m.strip() for m in gemini_fallback_env.split(",") if m.strip()]
            if gemini_fallback_env
            else list(self.GEMINI_FALLBACKS)
        )
        self._gemini_fallbacks = [m for m in configured_gemini_fallbacks if m and m != self.gemini_model]
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
        trimmed = text.strip() if text else ""
        if not trimmed:
            logger.log_ai_event("Empty or whitespace AI response; skipping", radio)
            return None
        # Drop obvious empty wrappers
        if trimmed in {"''", '""', "``", "()", "[]", "{}", "\"", "'"}:
            logger.log_ai_event("Only punctuation/quotes returned; treating as empty", radio)
            return None
        normalized = trimmed.lower()
        # Treat common "empty string" explanations as no-codeword
        if "empty string" in normalized or "no code" in normalized or "no keyword" in normalized:
            logger.log_ai_event("Interpreting model message as empty response", radio)
            return ""
        # this should be a keyword or codeword or so, not a long response. So filter out any long reply
        if len(trimmed) > 30:
            logger.log_ai_event("This is a way too long response to be a codeword, so skipping", radio)
            return None
        return trimmed

    def _is_rate_limit_error(self, error: Exception) -> bool:
        message = str(error).upper()
        return "RESOURCE_EXHAUSTED" in message or "429" in message or "RATE LIMIT" in message

    def _should_switch_gemini_model(self, error: Exception) -> bool:
        if self._is_rate_limit_error(error):
            return True
        message = str(error).lower()
        return any(
            token in message
            for token in (
                "model_not_found",
                "model not found",
                "unsupported model",
                "permission",
                "forbidden",
                "not enabled",
            )
        )

    def _update_gemini_model(self, new_model: str):
        """Keep the active Gemini model and rotate previous into the fallback list."""
        previous = getattr(self, "gemini_model", "")
        if previous and previous != new_model and previous not in self._gemini_fallbacks:
            self._gemini_fallbacks.append(previous)
        self.gemini_model = new_model
        self._gemini_fallbacks = [m for m in self._gemini_fallbacks if m != new_model]

    def _generate_with_gemini(self, prompt: str, radio: str, max_output_tokens: int) -> tuple[str | None, bool]:
        if not self.gemini_client:
            return None, True  # allow fallback when Gemini is not configured
        gen_config = {"max_output_tokens": min(max_output_tokens, 64)}
        models_to_try = [self.gemini_model] + [m for m in self._gemini_fallbacks if m != self.gemini_model]
        for idx, model_name in enumerate(models_to_try):
            response = None
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=self.PRE_PROMPT + prompt,
                    generation_config=gen_config,
                )
            except Exception as e:
                # Older/alternate SDKs may not accept generation_config; retry without it
                if isinstance(e, TypeError) and "generation_config" in str(e):
                    try:
                        response = self.gemini_client.models.generate_content(
                            model=model_name,
                            contents=self.PRE_PROMPT + prompt,
                            config=gen_config,
                        )
                    except Exception as inner_e:
                        e = inner_e
                        response = None
                if response is None:
                    print(f"GenAIHandler generate error (model={model_name}): {e}")
                    logger.log_ai_event(f"GenAIHandler generate error (model={model_name}): {e}", radio)
                    if self._should_switch_gemini_model(e) and idx < len(models_to_try) - 1:
                        next_model = models_to_try[idx + 1]
                        logger.log_ai_event(f"Switching Gemini model to fallback {next_model}", radio)
                        print(f"Switching Gemini model to fallback {next_model}")
                        continue
                    return None, True
            text = getattr(response, "text", "") or ""
            if model_name != self.gemini_model:
                logger.log_ai_event(f"Using Gemini fallback (model={model_name})", radio)
                print(f"Using Gemini fallback model {model_name}")
            self._update_gemini_model(model_name)
            return self._log_and_validate(text, radio), False
        return None, True

    def _generate_with_groq(self, prompt: str, radio: str, max_output_tokens: int) -> str | None:
        if not self.groq_client:
            logger.log_ai_event("Groq requested but not configured", radio)
            reason = self.groq_client_error or "unknown reason"
            print(f"Groq client not configured; skipping Groq call. Reason: {reason}")
            return None
        messages = (
            [{"role": "system", "content": self.PRE_PROMPT}] if self.PRE_PROMPT else []
        ) + [
            {"role": "user", "content": prompt},
            # Few-shot the empty-response expectation to reduce false positives (matches playground pattern)
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]
        params = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": 0,
        }
        params["max_tokens"] = min(max_output_tokens, 64)

        try:
            response = self.groq_client.chat.completions.create(**params)
            text = (response.choices[0].message.content or "").strip() if response and response.choices else ""
            if not text and self._groq_fallbacks:
                next_model = self._groq_fallbacks.pop(0)
                logger.log_ai_event(f"Groq returned empty response; switching model to {next_model}", radio)
                self.groq_model = next_model
                return self._generate_with_groq(prompt, radio, max_output_tokens)
            logger.log_ai_event(f"Using Groq fallback (model={self.groq_model})", radio)
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
            logger.log_ai_event("Falling back to Groq after Gemini issue", radio)
            print("Gemini issue detected; falling back to Groq")
            return self._generate_with_groq(prompt, radio, max_output_tokens)
        return None

if __name__ == "__main__":
    genai_handler = GenAIHandler(gemini_api_key="MY_API_KEY")
    test_prompt = "Detects that to us right now. You couldn't win a family four pack to six flags great America. Grizzly 408 516 1065 We got all Navy get 50 percent"

    print(genai_handler.generate(test_prompt))

    test_prompt2 = "long. All you need to do to qualify is text this word to us right now. What is demon as a demon drop? D-E-M-O-N. That's a spell. Demon. Text that to us right now.  Hexat to us right now and you and the family could be checking out six flags great America That's a 408 506 1065 looking for the top sales at Rayleigh's and I'm"
    print(genai_handler.generate(test_prompt2))
