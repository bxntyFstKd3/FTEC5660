#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_deepseek import ChatDeepSeek

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Read one Hong Kong supermarket receipt carefully.
Extract these fields:
- subtotal: the SUBTOTAL after discounts, before ROUNDING. If it is not
  printed, infer it from the final payment and the rounding adjustment.
- rounding: the signed ROUNDING adjustment (use 0.00 if absent).
- discounts: one positive HKD amount for each promotion, coupon, member,
  app, packaging-damage, percentage, or other discount that reduced the
  bill. For percentage discounts use the monetary reduction, not the
  percentage. Include each reduction once, even if printed as negative.
  Ignore ROUNDING, cash tendered, change, and loyalty points. Ignore a
  savings summary if it duplicates detailed discounts; use it if it is
  the only record of a discount.
Return only valid JSON with keys subtotal, rounding, discounts.
Use decimal strings without currency symbols for subtotal and rounding,
and an array of decimal strings for discounts. No explanation."""),
        MessagesPlaceholder("receipt_message"),
    ])

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        timeout=60,
        max_retries=1,
    )
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    def money(value: Any) -> Decimal:
        cleaned = (str(value).strip().replace(",", "")
                   .replace("HK$", "").replace("$", "")
                   .replace("−", "-"))
        return Decimal(cleaned)

    requests = [
        {"receipt_message": [HumanMessage(content=[
            {"type": "text", "text": "Extract the receipt fields as JSON."},
            {"type": "image_url", "image_url": {"url": image_data_url(path)}},
        ])]}
        for path in images
    ]
    replies = chain.batch(requests, config={"max_concurrency": 3})

    total_paid = Decimal("0.00")
    total_without_discount = Decimal("0.00")

    for path, reply in zip(images, replies):
        match = re.search(r"\{.*\}", response_text(reply), re.DOTALL)
        if match is None:
            raise ValueError(f"No JSON returned for {path.name}")

        fields = json.loads(match.group())
        subtotal = money(fields["subtotal"])
        rounding = money(fields["rounding"])
        discounts = fields["discounts"]

        if not isinstance(discounts, list):
            raise ValueError(f"Invalid discounts for {path.name}")

        total_paid += subtotal + rounding
        total_without_discount += subtotal + sum(
            (abs(money(value)) for value in discounts), Decimal("0.00")
        )

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }





# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
