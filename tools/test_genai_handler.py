import argparse
import os

from dotenv import load_dotenv

from utils.genai import GenAIHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick sanity check for GenAIHandler")
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "groq"],
        default="auto",
        help="Which provider to hit (auto tries Gemini then Groq on rate limits)",
    )
    parser.add_argument(
        "--prompt",
        default="The code word is PINEAPPLE. Please extract it.",
        help="Prompt/transcript to send",
    )
    parser.add_argument(
        "--pre-prompt",
        default="pre_prompt.txt",
        help="Path to pre-prompt file (optional)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens to request from the model",
    )
    parser.add_argument(
        "--radio",
        default="TEST",
        help="Radio tag for logging context",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics (env presence, pre-prompt info)",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    pre_prompt = args.pre_prompt if os.path.exists(args.pre_prompt) else ""
    has_groq_key = bool(os.getenv("GROQ_API_KEY"))
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))

    if args.verbose:
        print(f"Env: GROQ key present={has_groq_key}, GEMINI key present={has_gemini_key}")
        print(f"Pre-prompt path: {pre_prompt or '(none)'}")
        if pre_prompt:
            try:
                with open(pre_prompt, "r") as f:
                    content = f.read(120)
                print(f"Pre-prompt preview (first 120 chars): {content!r}")
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"Could not read pre-prompt: {exc}")
        print(f"Prompt: {args.prompt!r}")

    handler = GenAIHandler(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        pre_prompt_file=pre_prompt,
        provider=args.provider,
    )

    if args.verbose:
        groq_client = getattr(handler, "groq_client", None)
        gemini_client = getattr(handler, "gemini_client", None)
        groq_model = getattr(handler, "groq_model", "")
        remaining_fallbacks = getattr(handler, "_groq_fallbacks", [])
        print(f"Handler provider={handler.provider}, groq_client={'yes' if groq_client else 'no'} (model={groq_model or 'n/a'}, fallbacks={remaining_fallbacks}), gemini_client={'yes' if gemini_client else 'no'}")
        if not groq_client:
            reason = getattr(handler, "groq_client_error", "")
            print(f"Groq client not configured because: {reason or 'unknown'}")

    print(f"Using provider={args.provider}, pre_prompt={'yes' if pre_prompt else 'no'}")
    result = handler.generate(args.prompt, radio=args.radio, max_output_tokens=args.max_tokens)
    if args.verbose and handler.provider == "auto":
        print("Auto provider flow complete (Gemini with Groq fallback if needed).")
    print(f"Result: {result!r}")


if __name__ == "__main__":
    main()
