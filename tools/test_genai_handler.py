import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.genai import GenAIHandler

SAMPLE_CASES = [
    {
        "radio": "ALICE",
        "context": "have a fall out boy tickets all day today right now we're going to give you a code word mason's got one for you at 1205 and it's jane Jane has one at 505. You take these code words and head to Radialis.com That's our website and you can enter it in there and you'll be in the running for this. These tickets are incredible This hours code This hour's code word is MUSIC, M-U-S-I-C. Take the word MUSIC to Radialis.com to enter for tickets to fall.",
        "expected": "music",
    },
    {
        "radio": "MIX106.5",
        "context": "You already know Angela Johnson being back in the bay Even the Great Mall Nail Solange are like, oh no! She's watching us! Hahaha, ya saw me! Okay, if you want to win these tickets your code word is lol because that's what you gonna be doing a lot of lol of LOL-ing, LOL, that's your code word, LOL. 408-516-1065, text that code word.",
        "expected": "lol",
    },
    {
        "radio": "ALICE",
        "context": "at the Masonic Dude. This is one of the most talked about albums of the year. Like, and she's only doing nine shows in all of. in all of North America, you could see it with the code word POP, P-O-P, for your chance to see Lily. see lily allen doing west Put it, your Lily Allen code word is POP. P-O-P.",
        "expected": "pop",
    },
    {
        "radio": "ALICE",
        "context": "Thanks for watching! All this week, Alice 97.3 is stuffing your stocking with some of the biggest shows of 2026. Hey Alice listeners, you don't want to miss this Olivia Dean ariana grande Louis Capaldi and the San Francisco Symphony. Listen for the code word at 5 past the hours of 9 a.m., noon, and 5 p.m. Enter the-",
        "expected": "",
    },
    {
        "radio": "ALT94.7",
        "context": "you've been so good this year we'd like to give you the gift of concert of concert tickets Santa all 94 seven is going through this year In this year, with aftershock respect stuck in your stocking, off from gettin' cold, that's a whole lotta metal! Listen all this- Listen all this week at 9, noon four and seven for your chance to win four day general mission response before the lineup is even announced. Full detail.",
        "expected": "",
    },
    {
        "radio": "ALICE",
        "context": "Alice 97 3 is stuffing your stocking with some of the biggest shows of 2026 Hey Alice listeners, you don't want to miss this. Olivia Dean, Ariana Grande, Hey there Aria! Louis Capaldi, and the Sam Prince. San Francisco Symphony. Listen for the code word at five past the hours of 9am, noon and 5pm. Enter that at RadioEllis.com. For your chance to win tickets to all of these amazing shows. Courtesy of Messina touring group Golden Voice.",
        "expected": "",
    },
    {
        "radio": "MIX106.5",
        "context": "Oh, and guess what? Bluey and Bingo are coming for real life. Make memories that last a lifetime. during the 70th celebration! From Paint the Night Parade returning January 30th to World of Color Happiness, and with... and with Bluey and Bingo coming soon, happiness is everywhere at the Disneyland Resort. Save more than 50% when you purchase... When you purchase the 3-day California Resident Park Hopper ticket, total price $249, valid January 1st through May 21st, 2020. 2021-2026. Visit Disneyland.com for details. Savings based on regular priced adult 3-day park hopper ticket. Cashback is not available on gas in New Jersey, Wisconsin. $5,000. That's the average amount of money people in the U.S. are now spending on...",
        "expected": "",
    },
    {
        "radio": "ALT94.7",
        "context": "I can't guarantee sunshine, but if you've been missing the rain amongst all this Misty Fog. That's supposed to return later in the week, I'm Dallas. Before we get to all that, more of the same For more of the same type of weather we've been having in a 94 minute commercial free music stretch approaching about 7.5. 715 then another chance at those four-day wristbands to Aftershock at 9 a.m. It's all 94-7",
        "expected": "",
    },
    {
        "radio": "ALICE",
        "context": "And that fallout boy code word is coming up at 505. This is the show I've been talking about that's at that incredibly incredibly intimate venue for them, the Regency Ballroom. It's headed into big game weekend and tickets aren't even on sale. on sale for the big, massive, general public. Tickets are only on sale for this Fall Out Boy show for Wells Fargo. Wells Fargo credit card holders. But again, I have your chance to win tickets. Your Fallout Boy code word coming up at 505.",
        "expected": "",
    },
    {
        "radio": "ALICE",
        "context": "97.3 is loading up the sleigh with tickets to see phone out for Some of them that don't stop to go Have you heard any fifth at the Regency Ballroom? Hey, this is Fall Out Boy. See us live. Listen for- Listen for the code word at five past the hours of 9 a.m. Noon and 5 p.m. Then enter that at RadioAlice.com for your tan For your chance to go to the show, get everything Fall Out Boy at RadioAlice.com Hi, I'm Tarek Kabis. We have a real...",
        "expected": "",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick sanity check for GenAIHandler")
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "groq", "mistral"],
        default="auto",
        help="Which provider to hit (auto interleaves Gemini -> Mistral -> Groq primaries, then their fallbacks in that order)",
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
    parser.add_argument(
        "--run-samples",
        action="store_true",
        help="Run the built-in sample contexts instead of the single prompt",
    )
    parser.add_argument(
        "--sample-file",
        default="",
        help="Optional JSON file to override the sample set (list of {radio, context, expected})",
    )
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated 1-based sample numbers to run with --run-samples (e.g. '1,3,5')",
    )
    return parser.parse_args()


def _load_sample_cases(sample_file: str | None) -> list[dict]:
    if sample_file:
        try:
            with open(sample_file, "r") as f:
                data = json.load(f)
            cases = []
            for idx, entry in enumerate(data):
                if not isinstance(entry, dict):
                    continue
                ctx = (entry.get("context") or "").strip()
                if not ctx:
                    continue
                cases.append(
                    {
                        "radio": (entry.get("radio") or "").strip() or "TEST",
                        "context": ctx,
                        "expected": (entry.get("expected") or "").strip(),
                    }
                )
            if cases:
                print(f"Loaded {len(cases)} sample cases from {sample_file}")
                return cases
            print(f"No usable cases found in {sample_file}; falling back to built-in samples.")
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"Could not read sample file {sample_file}: {exc}")
    return SAMPLE_CASES


def _run_sample_batch(handler: GenAIHandler, cases: list[dict], max_tokens: int, selected_indices: list[int] | None = None):
    if selected_indices:
        selected_cases = []
        for idx in selected_indices:
            if 1 <= idx <= len(cases):
                selected_cases.append((idx, cases[idx - 1]))
            else:
                print(f"Ignoring sample index {idx}: out of range (1-{len(cases)})")
    else:
        selected_cases = list(enumerate(cases, start=1))

    total = len(selected_cases)
    failures = []
    print(f"Running {total} sample cases...")
    for run_idx, (orig_idx, case) in enumerate(selected_cases, start=1):
        radio = case.get("radio", "TEST")
        context = case.get("context", "")
        expected = (case.get("expected") or "").strip()
        print(f"[{run_idx}/{total}] sample#{orig_idx} radio={radio} expected='{expected or '(none)'}'")
        result = handler.generate(context, radio=radio, max_output_tokens=max_tokens)
        found = (result or "").strip() if result else ""
        match = (found.lower() == expected.lower()) if expected else (found == "")
        provider = handler.last_provider or "n/a"
        model = handler.last_model or "n/a"
        status = "OK" if match else "FAIL"
        print(f"  -> got='{found or '(empty)'}' [{status}] provider={provider}, model={model}")
        if not match:
            failures.append(
                {
                    "index": orig_idx,
                    "radio": radio,
                    "expected": expected,
                    "found": found,
                    "provider": provider,
                    "model": model,
                }
            )

    print(f"Finished samples: {total - len(failures)} passed / {total} total.")
    if failures:
        print("Failures:")
        for item in failures:
            print(
                f"  #{item['index']} radio={item['radio']} expected='{item['expected'] or '(none)'}' got='{item['found'] or '(empty)'}' provider={item['provider']} model={item['model']}"
            )


def _print_handler_info(handler: GenAIHandler):
    groq_client = getattr(handler, "groq_client", None)
    gemini_client = getattr(handler, "gemini_client", None)
    mistral_client = getattr(handler, "mistral_client", None)
    groq_model = getattr(handler, "groq_model", "")
    gemini_model = getattr(handler, "gemini_model", "")
    mistral_model = getattr(handler, "mistral_model", "")
    remaining_fallbacks = getattr(handler, "_groq_fallbacks", [])
    gemini_fallbacks = getattr(handler, "_gemini_fallbacks", [])
    mistral_fallbacks = getattr(handler, "_mistral_fallbacks", [])
    print(f"Handler provider={handler.provider}, groq_client={'yes' if groq_client else 'no'} (model={groq_model or 'n/a'}, fallbacks={remaining_fallbacks}), gemini_client={'yes' if gemini_client else 'no'} (model={gemini_model or 'n/a'}, fallbacks={gemini_fallbacks}), mistral_client={'yes' if mistral_client else 'no'} (model={mistral_model or 'n/a'}, fallbacks={mistral_fallbacks})")
    if not groq_client:
        reason = getattr(handler, "groq_client_error", "")
        print(f"Groq client not configured because: {reason or 'unknown'}")
    if not mistral_client:
        reason = getattr(handler, "mistral_client_error", "")
        print(f"Mistral client not configured because: {reason or 'unknown'}")


def main():
    load_dotenv()
    args = parse_args()

    pre_prompt = args.pre_prompt if os.path.exists(args.pre_prompt) else ""
    has_groq_key = bool(os.getenv("GROQ_API_KEY"))
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    has_mistral_key = bool(os.getenv("MISTRAL_API_KEY"))

    if args.verbose:
        print(f"Env: GROQ key present={has_groq_key}, GEMINI key present={has_gemini_key}, MISTRAL key present={has_mistral_key}")
        print(f"Pre-prompt path: {pre_prompt or '(none)'}")
        if pre_prompt:
            try:
                with open(pre_prompt, "r") as f:
                    content = f.read(120)
                print(f"Pre-prompt preview (first 120 chars): {content!r}")
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"Could not read pre-prompt: {exc}")
        if not args.run_samples:
            print(f"Prompt: {args.prompt!r}")

    handler = GenAIHandler(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
        pre_prompt_file=pre_prompt,
        provider=args.provider,
    )

    if args.verbose:
        _print_handler_info(handler)

    if args.run_samples:
        cases = _load_sample_cases(args.sample_file)
        selected_indices = None
        if args.sample_ids:
            try:
                selected_indices = [
                    int(part)
                    for part in args.sample_ids.replace(" ", "").split(",")
                    if part
                ]
            except ValueError:
                print(f"Could not parse --sample-ids '{args.sample_ids}'; running all samples.")
                selected_indices = None
        _run_sample_batch(handler, cases, args.max_tokens, selected_indices)
        return

    if args.verbose:
        _print_handler_info(handler)

    print(f"Using provider={args.provider}, pre_prompt={'yes' if pre_prompt else 'no'}")
    result = handler.generate(args.prompt, radio=args.radio, max_output_tokens=args.max_tokens)
    if args.verbose and handler.provider == "auto":
        print("Auto provider flow complete (Gemini -> Groq -> Mistral fallbacks as needed).")
    used_provider = handler.last_provider or "(unknown)"
    used_model = handler.last_model or "(unknown)"
    print(f"AI used: provider={used_provider}, model={used_model}")
    print(f"Result: {result!r}")


if __name__ == "__main__":
    main()
