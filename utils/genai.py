import os
import time
import json
import re

import google.genai as genai
from utils import logger

try:
    from groq import Groq
except ImportError:  # pragma: no cover - safety if dependency missing
    Groq = None

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover - safety if dependency missing
    Mistral = None


DEFAULT_JSON_RESULT = {"codeword": None, "confidence": 0, "evidence": None}


def extract_first_json_object(text: str) -> str | None:
    """
    Scan the text for the first balanced JSON object starting at the first '{'.
    Returns the raw substring including braces, or None if not found.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_or_reject(raw_text: str) -> dict:
    """
    Strictly parse model output into the required schema.
    Falls back to balanced-brace extraction before rejecting.
    """
    def _default():
        return dict(DEFAULT_JSON_RESULT)

    def _clean_string(val):
        if val is None:
            return None
        if not isinstance(val, str):
            return None
        stripped = val.strip()
        return stripped if stripped else None

    candidate_text = raw_text or ""
    candidate_text = candidate_text.strip()
    # Strip common code-fence wrappers (e.g., ```json ... ```)
    if candidate_text.startswith("```"):
        candidate_text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", candidate_text).strip()
    obj = None
    try:
        obj = json.loads(candidate_text)
    except Exception:
        extracted = extract_first_json_object(candidate_text)
        if not extracted:
            return _default()
        try:
            obj = json.loads(extracted)
        except Exception:
            return _default()

    if not isinstance(obj, dict):
        return _default()

    if set(obj.keys()) != {"codeword", "confidence", "evidence"}:
        return _default()

    codeword = _clean_string(obj.get("codeword"))
    evidence = _clean_string(obj.get("evidence"))
    confidence = obj.get("confidence")
    try:
        confidence_val = float(confidence)
    except Exception:
        return _default()

    if confidence_val < 0 or confidence_val > 1:
        return _default()

    if evidence is not None and codeword is None:
        # If no codeword, evidence must be null
        return _default()

    if codeword is None:
        return {
            "codeword": None,
            "confidence": confidence_val,
            "evidence": None,
        }

    if evidence is None:
        return _default()

    return {
        "codeword": codeword,
        "confidence": confidence_val,
        "evidence": evidence,
    }


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
    MISTRAL_MODEL = "mistral-large-latest"
    MISTRAL_FALLBACKS = [
        "mistral-small-latest",
        "open-mixtral-8x7b",
    ]
    COOLDOWN_SECONDS = 3600  # one hour backoff after 429/rate limit

    def __init__(
        self,
        gemini_api_key: str = "",
        pre_prompt_file: str = "",
        groq_api_key: str = "",
        mistral_api_key: str = "",
        provider: str = "auto",
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
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
        mistral_env_model = os.getenv("MISTRAL_MODEL", "").strip()
        self.mistral_model = mistral_env_model or self.MISTRAL_MODEL
        mistral_fallback_env = os.getenv("MISTRAL_FALLBACKS", "")
        configured_mistral_fallbacks = (
            [m.strip() for m in mistral_fallback_env.split(",") if m.strip()]
            if mistral_fallback_env
            else list(self.MISTRAL_FALLBACKS)
        )
        self._mistral_fallbacks = [m for m in configured_mistral_fallbacks if m and m != self.mistral_model]
        self.mistral_client_error = ""
        if self.mistral_api_key:
            if Mistral is None:
                self.mistral_client = None
                self.mistral_client_error = "mistralai SDK missing; install/upgrade via `pip install -U mistralai`"
            else:
                try:
                    self.mistral_client = Mistral(api_key=self.mistral_api_key)
                except Exception as exc:  # pragma: no cover - defensive
                    self.mistral_client = None
                    self.mistral_client_error = f"Failed to init Mistral client: {exc}"
        else:
            self.mistral_client = None
            self.mistral_client_error = "No Mistral API key provided."
        provider = (provider or "auto").lower()
        self.provider = provider if provider in ("auto", "gemini", "groq", "mistral") else "auto"
        self.pre_prompt_file = pre_prompt_file
        self._pre_prompt_mtime = None
        self.PRE_PROMPT = ""
        self.last_provider = ""
        self.last_model = ""
        self.last_confidence: float = 0.0
        self.last_evidence: str | None = None
        self.last_result: dict = dict(DEFAULT_JSON_RESULT)
        self._model_cooldowns = {}
        self._load_pre_prompt()
        self._init_model_cooldowns()

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

    def _init_model_cooldowns(self):
        all_models = set()
        all_models.add(self.gemini_model)
        all_models.update(self._gemini_fallbacks)
        all_models.add(self.groq_model)
        all_models.update(self._groq_fallbacks)
        all_models.add(self.mistral_model)
        all_models.update(self._mistral_fallbacks)
        for model in all_models:
            self._model_cooldowns[model] = 0.0

    def _model_in_cooldown(self, model: str) -> bool:
        until = self._model_cooldowns.get(model, 0.0)
        return until and time.time() < until

    def _set_model_cooldown(self, provider: str, model: str, radio: str, reason: str):
        self._model_cooldowns[model] = time.time() + self.COOLDOWN_SECONDS
        logger.log_ai_event(
            f"{provider} model {model} hit rate limit; cooling down for {self.COOLDOWN_SECONDS} seconds (reason: {reason})",
            radio,
        )
        print(f"{provider} model {model} entered cooldown for {self.COOLDOWN_SECONDS}s because: {reason}")

    def _log_and_validate(self, text: str, radio: str = "") -> tuple[str | None, str | None, bool]:
        parsed = parse_json_or_reject(text)
        self.last_result = dict(parsed)
        try:
            self.last_confidence = float(parsed.get("confidence", 0) or 0)
        except Exception:
            self.last_confidence = 0.0
        self.last_evidence = parsed.get("evidence")
        codeword = parsed.get("codeword")
        confidence = self.last_confidence
        evidence = self.last_evidence

        if codeword is None:
            logger.log_ai_event(f"Response: <empty> (confidence={confidence})", radio)
            if text and parsed == DEFAULT_JSON_RESULT:
                logger.log_ai_event("Parsed to default/null despite non-empty raw response", radio)
            return "", None, False

        logger.log_ai_event(
            f"Response: codeword={codeword} confidence={confidence} evidence={evidence}",
            radio,
        )
        return codeword, None, False

    def _is_rate_limit_error(self, error: Exception) -> bool:
        message = str(error).upper()
        return "RESOURCE_EXHAUSTED" in message or "429" in message or "RATE LIMIT" in message

    def _should_switch_gemini_model(self, error: Exception) -> bool:
        if self._is_rate_limit_error(error):
            return True
        message = str(error).lower()
        return any(token in message for token in self._model_not_found_tokens())

    @staticmethod
    def _model_not_found_tokens():
        return (
            "model_not_found",
            "model not found",
            "unsupported model",
            "permission",
            "forbidden",
            "not enabled",
            "decommissioned",
            "no longer supported",
        )

    def _provider_models(self, provider: str) -> list[str]:
        if provider == "gemini":
            return [self.gemini_model] + [m for m in self._gemini_fallbacks if m]
        if provider == "mistral":
            return [self.mistral_model] + [m for m in self._mistral_fallbacks if m]
        if provider == "groq":
            return [self.groq_model] + [m for m in self._groq_fallbacks if m]
        return []

    def _record_model_usage(self, provider: str, model: str, radio: str):
        """Persist the last successful provider/model and emit a log line."""
        self.last_provider = (provider or "").lower()
        self.last_model = model or ""
        provider_label = self.last_provider or "unknown"
        model_label = self.last_model or "unknown"
        logger.log_ai_event(f"AI choice: provider={provider_label}, model={model_label}", radio)

    def _call_model(self, provider: str, model: str, prompt: str, radio: str, max_output_tokens: int) -> tuple[str | None, str | None]:
        """
        Execute a single model call and return (validated_text, error_reason).
        error_reason is a human-readable string when the model call failed or produced unusable output.
        """
        requested_tokens = max(128, min(max_output_tokens, 256))

        if provider == "gemini":
            if not self.gemini_client:
                return None, "gemini client not configured"
            gen_config = {"max_output_tokens": requested_tokens, "temperature": 0}
            response = None
            try:
                # Newer SDK signature
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=self.PRE_PROMPT + prompt,
                    generation_config=gen_config,
                    response_mime_type="application/json",
                )
            except Exception:
                pass
            if response is None:
                try:
                    # Older signature: config + response_mime_type
                    response = self.gemini_client.models.generate_content(
                        model=model,
                        contents=self.PRE_PROMPT + prompt,
                        config=gen_config,
                        response_mime_type="application/json",
                    )
                except Exception:
                    pass
            if response is None:
                try:
                    # Older signature: generation_config without response_mime_type
                    response = self.gemini_client.models.generate_content(
                        model=model,
                        contents=self.PRE_PROMPT + prompt,
                        generation_config=gen_config,
                    )
                except Exception:
                    pass
            if response is None:
                try:
                    # Minimal signature: contents only
                    response = self.gemini_client.models.generate_content(
                        model=model,
                        contents=self.PRE_PROMPT + prompt,
                    )
                except Exception as e:
                    logger.log_ai_event(f"GenAIHandler generate error (model={model}): {e}", radio)
                    print(f"GenAIHandler generate error (model={model}): {e}")
                    if self._is_rate_limit_error(e):
                        self._set_model_cooldown(provider, model, radio, str(e))
                    else:
                        self._set_model_cooldown(provider, model, radio, f"error: {e}")
                    return None, str(e)
            text = getattr(response, "text", "") if response is not None else ""
            if not text:
                logger.log_ai_event("Gemini returned empty content", radio)
            else:
                logger.log_ai_event(f"Gemini raw: {text}", radio)
            validated, reason, cooldown = self._log_and_validate(text, radio)
            if validated is None:
                if reason and cooldown:
                    self._set_model_cooldown(provider, model, radio, reason)
                return None, reason or "empty or invalid response"
            return validated, None
            text = getattr(response, "text", "") or ""
            validated, reason, cooldown = self._log_and_validate(text, radio)
            if validated is None:
                if reason and cooldown:
                    self._set_model_cooldown(provider, model, radio, reason)
                return None, reason or "empty or invalid response"
            return validated, None

        if provider == "groq":
            if not self.groq_client:
                reason = self.groq_client_error or "groq client not configured"
                logger.log_ai_event("Groq requested but not configured", radio)
                print(f"Groq client not configured; skipping Groq call. Reason: {reason}")
                return None, reason
            messages = (
                [{"role": "system", "content": self.PRE_PROMPT}] if self.PRE_PROMPT else []
            ) + [
                {"role": "user", "content": prompt},
            ]
            params = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": requested_tokens,
                "response_format": {"type": "json_object"},
            }
            try:
                response = self.groq_client.chat.completions.create(**params)
                text = (response.choices[0].message.content or "").strip() if response and response.choices else ""
                if not text:
                    logger.log_ai_event("Groq returned empty content", radio)
                validated, reason, cooldown = self._log_and_validate(text, radio)
                if validated is None:
                    if reason and cooldown:
                        self._set_model_cooldown(provider, model, radio, reason)
                    return None, reason or "empty or invalid response"
                return validated, None
            except Exception as e:
                print(f"Groq generate error: {e}")
                logger.log_ai_event(f"Groq generate error: {e}", radio)
                if self._is_rate_limit_error(e):
                    self._set_model_cooldown(provider, model, radio, str(e))
                else:
                    self._set_model_cooldown(provider, model, radio, f"error: {e}")
                return None, str(e)

        if provider == "mistral":
            if not self.mistral_client:
                reason = self.mistral_client_error or "mistral client not configured"
                logger.log_ai_event("Mistral requested but not configured", radio)
                print(f"Mistral client not configured; skipping Mistral call. Reason: {reason}")
                return None, reason
            messages = (
                [{"role": "system", "content": self.PRE_PROMPT}] if self.PRE_PROMPT else []
            ) + [
                {"role": "user", "content": prompt},
            ]
            params = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": requested_tokens,
            }
            try:
                response = self.mistral_client.chat.complete(**params)
                text = ""
                try:
                    choice = response.choices[0] if response and getattr(response, "choices", None) else None
                    message = getattr(choice, "message", None) if choice else None
                    content = getattr(message, "content", "") if message else ""
                    if isinstance(content, list):  # mistralai may return list of dicts
                        text = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                        )
                    else:
                        text = content or ""
                    text = text.strip()
                except Exception:
                    text = ""
                if not text:
                    logger.log_ai_event("Mistral returned empty content", radio)
                else:
                    logger.log_ai_event(f"Mistral raw: {text}", radio)
                validated, reason, cooldown = self._log_and_validate(text, radio)
                if validated is None:
                    if reason and cooldown:
                        self._set_model_cooldown(provider, model, radio, reason)
                    return None, reason or "empty or invalid response"
                return validated, None
            except Exception as e:
                print(f"Mistral generate error: {e}")
                logger.log_ai_event(f"Mistral generate error: {e}", radio)
                if self._is_rate_limit_error(e):
                    self._set_model_cooldown(provider, model, radio, str(e))
                else:
                    self._set_model_cooldown(provider, model, radio, f"error: {e}")
                return None, str(e)

        return None, "unknown provider"

    def _run_provider_models(self, provider: str, prompt: str, radio: str, max_output_tokens: int) -> str | None:
        models = self._provider_models(provider)
        for idx, model in enumerate(models):
            if self._model_in_cooldown(model):
                logger.log_ai_event(f"{provider} model {model} is in cooldown; skipping", radio)
                print(f"{provider} model {model} in cooldown; skipping")
                continue
            result, error = self._call_model(provider, model, prompt, radio, max_output_tokens)
            if result is not None:
                if model != models[0]:
                    logger.log_ai_event(f"Using {provider} fallback (model={model})", radio)
                    print(f"Using {provider} fallback model {model}")
                self._record_model_usage(provider, model, radio)
                return result
            if error and idx < len(models) - 1:
                next_model = models[idx + 1]
                logger.log_ai_event(f"{provider} model {model} failed; trying fallback {next_model} because: {error}", radio)
                print(f"{provider} model {model} failed; trying fallback {next_model} because: {error}")
            elif error:
                logger.log_ai_event(f"{provider} model {model} failed with no further fallbacks: {error}", radio)
                print(f"{provider} model {model} failed with no further fallbacks: {error}")
        return None


    def generate(self, prompt: str, radio: str = "", max_output_tokens: int = 1024) -> str | None:
        # Reset last-used metadata for this call to avoid stale values when generation fails.
        self.last_provider = ""
        self.last_model = ""
        self.last_confidence = 0.0
        self.last_evidence = None
        self.last_result = dict(DEFAULT_JSON_RESULT)
        self._check_pre_prompt_update()
        logger.log_ai_event(f"Context: {prompt}", radio)
        if self.provider == "groq":
            return self._run_provider_models("groq", prompt, radio, max_output_tokens)

        if self.provider == "gemini":
            return self._run_provider_models("gemini", prompt, radio, max_output_tokens)

        if self.provider == "mistral":
            return self._run_provider_models("mistral", prompt, radio, max_output_tokens)

        # auto: sequential provider order gemini -> mistral -> groq
        for provider in ("gemini", "mistral", "groq"):
            text = self._run_provider_models(provider, prompt, radio, max_output_tokens)
            if text is not None:
                return text
        return None

if __name__ == "__main__":
    genai_handler = GenAIHandler(gemini_api_key="MY_API_KEY")
    test_prompt = "Detects that to us right now. You couldn't win a family four pack to six flags great America. Grizzly 408 516 1065 We got all Navy get 50 percent"

    print(genai_handler.generate(test_prompt))

    test_prompt2 = "long. All you need to do to qualify is text this word to us right now. What is demon as a demon drop? D-E-M-O-N. That's a spell. Demon. Text that to us right now.  Hexat to us right now and you and the family could be checking out six flags great America That's a 408 506 1065 looking for the top sales at Rayleigh's and I'm"
    print(genai_handler.generate(test_prompt2))
