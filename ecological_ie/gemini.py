import io
import json
import re
import threading
import time
from pathlib import Path

from PIL import Image

from google import genai
from google.genai import types

from .config import PRICES_PER_M, Settings

MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".tif": "image/tiff",
              ".tiff": "image/tiff", ".webp": "image/webp"}
IMAGE_RESOLUTIONS = {
    "low": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
    "medium": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
    "high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
    "ultra_high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH,
}


def response_text(response) -> str:
    if getattr(response, "text", None):
        return response.text
    for candidate in getattr(response, "candidates", None) or []:
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        texts = [part.text for part in parts if getattr(part, "text", None)]
        if texts:
            return "".join(texts)
    return ""


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for i, char in enumerate(text):
            if char in "{[":
                try:
                    return decoder.raw_decode(text[i:])[0]
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON object found in response")


def usage_of(response) -> dict:
    metadata = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": getattr(metadata, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(metadata, "candidates_token_count", 0) or 0,
        "thinking_tokens": getattr(metadata, "thoughts_token_count", 0) or 0,
    }


def cost_usd(usage: dict, settings: Settings) -> float:
    price_in, price_out = PRICES_PER_M.get(usage.get("model") or settings.model, (2.00, 12.00))
    output = usage.get("output_tokens", 0) + usage.get("thinking_tokens", 0)
    return usage.get("input_tokens", 0) / 1e6 * price_in + output / 1e6 * price_out


class Gemini:
    def __init__(self, api_key: str, settings: Settings):
        if not api_key:
            raise RuntimeError("No Gemini API key: set GEMINI_API_KEY / GOOGLE_API_KEY or put it into .env")
        self.client = genai.Client(api_key=api_key)
        self.settings = settings
        self.ledger: list[dict] = []
        self.ledger_lock = threading.Lock()

    def spent(self) -> float:
        with self.ledger_lock:
            return sum(entry["cost_usd"] for entry in self.ledger)

    def image_part(self, image, mime_type: str = "image/jpeg"):
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "JPEG", quality=92)
            image = buffer.getvalue()
        elif isinstance(image, (str, Path)):
            image = Path(image)
            mime_type = MIME_TYPES.get(image.suffix.lower(), mime_type)
            image = image.read_bytes()
        return types.Part.from_bytes(data=image, mime_type=mime_type,
                                     media_resolution=IMAGE_RESOLUTIONS[self.settings.image_resolution])

    def config(self, schema: dict | None, thinking: str | None = None):
        options = dict(
            max_output_tokens=self.settings.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=thinking or self.settings.thinking_level),
            response_mime_type="application/json",
        )
        if schema is not None:
            options["response_json_schema"] = schema
        return types.GenerateContentConfig(**options)

    def extract(self, prompt: str, schema: dict, images=(), model: str | None = None,
                thinking: str | None = None, prompt_after_images: str = "") -> tuple[dict, dict, str]:
        use_schema, last_error = True, None
        image_parts = [self.image_part(image) for image in images]
        trailer = [prompt_after_images] if prompt_after_images else []
        for attempt in range(self.settings.retries + 1):
            text = prompt if use_schema else (
                prompt + "\n\nReturn ONLY a JSON object that conforms to this JSON schema:\n"
                + json.dumps(schema, ensure_ascii=False))
            try:
                response = self.client.models.generate_content(
                    model=model or self.settings.model, contents=[text, *image_parts, *trailer],
                    config=self.config(schema if use_schema else None, thinking))
                answer = response_text(response)
                if not answer:
                    reason = getattr((response.candidates or [None])[0], "finish_reason", "?")
                    raise ValueError(f"empty response (finish_reason={reason})")
                usage = usage_of(response)
                usage["model"] = model or self.settings.model
                with self.ledger_lock:
                    self.ledger.append({**usage, "cost_usd": cost_usd(usage, self.settings)})
                return parse_json(answer), usage, "schema" if use_schema else "json_mime"
            except Exception as error:
                last_error = error
                message = str(error).lower()
                if use_schema and ("schema" in message or "invalid_argument" in message or "400" in message):
                    use_schema = False
                    print("  ! schema rejected, falling back to json_mime:", str(error)[:200])
                    continue
                wait = 2.0 * (attempt + 1)
                if "429" in message or "resource_exhausted" in message or "quota" in message:
                    wait = 15.0 * (attempt + 1)
                print(f"  ! attempt {attempt + 1} failed: {str(error)[:160]} – retry in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after {self.settings.retries + 1} attempts: {last_error}")
