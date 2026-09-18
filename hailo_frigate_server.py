import os
import sys
import base64
import io
import asyncio
import json
import traceback
import re
import ast
import time
import logging
import urllib.request
from contextlib import asynccontextmanager
from typing import List, Dict, Union, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request

try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    from pydantic import BaseModel, Field
    ConfigDict = None

try:
    from hailo_platform import VDevice
    from hailo_platform.genai import VLM
except ImportError as err:
    print(f"CRITICAL ENVIRONMENT ROADBLOCK: HailoRT Platform SDK missing from active environment: {err}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
HEF_MODEL_PATH = os.environ.get(
    "HEF_MODEL_PATH",
    "/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef",
)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen2-VL-2B-Instruct.hef")
TARGET_NPU_DIM = 336

# ---------------------------------------------------------------------------
# Logging Configuration
# All events are written to their respective log files in LOG_DIR.
# The booleans below control whether they are ALSO output to the console.
# ---------------------------------------------------------------------------
LOG_DIR = os.environ.get(
    "FRIGATE_LOG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
)

# Master switch: set to False to silence all console logging
LOG_TO_CONSOLE = True

# Granular console output toggles (only active when LOG_TO_CONSOLE is True):
LOG_CONSOLE_INCOMING_REQUEST = False  # Log incoming Frigate request JSON to console
LOG_CONSOLE_PROMPTS = True            # Log prompts sent to model to console
LOG_CONSOLE_MODEL_RAW = True          # Log raw VLM model output to console
LOG_CONSOLE_RESPONSES = True          # Log responses sent to Frigate to console
LOG_CONSOLE_TIMING = True             # Log request timing summary box to console
LOG_CONSOLE_QUEUE = True              # Log queue/timeout monitor events to console
LOG_CONSOLE_SERVER = True             # Log server startup/cache resets to console

# Backward-compatibility aliases
LOG_INCOMING_REQUEST = LOG_CONSOLE_INCOMING_REQUEST
LOG_FRIGATE_PROMPT = LOG_CONSOLE_PROMPTS


def log_event(filename: str, message: str, console_enabled: bool = True) -> None:
    """
    Appends the message to the specified log file in LOG_DIR.
    Optionally prints to console if both LOG_TO_CONSOLE and console_enabled are True.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, filename)
        clean_msg = str(message).strip()
        with open(filepath, "a", encoding="utf-8") as f:
            if (
                not clean_msg.startswith("[20")
                and "--- LOGGING" not in clean_msg
                and "REQUEST TIMING" not in clean_msg
            ):
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {clean_msg}\n\n")
            else:
                f.write(f"{clean_msg}\n\n")
    except Exception as err:
        print(f"[Logger Error] Failed writing to {filename}: {err}", flush=True)

    if LOG_TO_CONSOLE and console_enabled:
        print(message, flush=True)


DEFAULT_REVIEW_PROPERTIES = {
    "title": {"type": "string"},
    "scene": {"type": "string"},
    "shortSummary": {"type": "string"},
    "confidence": {"type": "number"},
    "other_concerns": {"type": "array", "items": {"type": "string"}},
    "potential_threat_level": {"type": "integer"},
    "observations": {"type": "array", "items": {"type": "string"}},
}

vdevice = None
vlm_instance = None
npu_lock = asyncio.Lock()
queued_requests_count = 0

TIMEOUT_LOG_FILE = os.environ.get(
    "FRIGATE_TIMEOUT_LOG",
    os.path.join(LOG_DIR, "frigate_timeouts.log"),
)

timeout_logger = logging.getLogger("frigate_timeouts")
timeout_logger.setLevel(logging.WARNING)
if not timeout_logger.handlers:
    os.makedirs(LOG_DIR, exist_ok=True)
    _file_handler = logging.FileHandler(TIMEOUT_LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    timeout_logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# Concurrency & Queue Monitoring
# ---------------------------------------------------------------------------
async def _cancel_and_release(task: asyncio.Task, lock: asyncio.Lock) -> None:
    task.cancel()
    try:
        await task
        lock.release()
    except (asyncio.CancelledError, Exception):
        pass


def log_timed_out_request(
    chat_request: Optional[Any],
    reason: str,
    wait_time: float,
    prompt_to_model: Optional[str] = None,
) -> None:
    if not prompt_to_model and chat_request:
        msg_text = extract_message_content_text(getattr(chat_request, "messages", []))
        condensed = condense_prompt(msg_text, max_chars=600) or "Analyze what is happening in this security camera frame."
        prompt_to_model = build_model_prompt_log(msg_text, condensed)

    if prompt_to_model:
        timeout_logger.warning(f"{reason} (waited {wait_time:.1f}s):\n{prompt_to_model}")
    else:
        timeout_logger.warning(f"Request dropped: {reason} (waited {wait_time:.1f}s)")


async def acquire_lock_or_abort(
    lock: asyncio.Lock,
    http_request: Request,
    chat_request: Optional[Any] = None,
    prompt_to_model: Optional[str] = None,
) -> bool:
    """Waits for the NPU lock, aborting only if the Frigate client cancels/disconnects."""
    global queued_requests_count

    was_queued = lock.locked()
    if was_queued:
        queued_requests_count += 1
        log_event(
            "queue_monitor.log",
            f"[Queue Monitor] Active request running on model. Request entered queue. Currently queued: {queued_requests_count}",
            console_enabled=LOG_CONSOLE_QUEUE,
        )

    acquire_task = asyncio.create_task(lock.acquire())
    disconnect_task = asyncio.create_task(http_request.is_disconnected())
    start_time = time.time()

    try:
        done, _ = await asyncio.wait(
            [acquire_task, disconnect_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if acquire_task in done:
            disconnect_task.cancel()
            return True

        # Client disconnected / cancelled while waiting in queue
        elapsed = time.time() - start_time
        msg = f"[Queue Monitor] Frigate client cancelled/disconnected while waiting in queue ({elapsed:.1f}s). Dropping request."
        log_event("queue_monitor.log", msg, console_enabled=LOG_CONSOLE_QUEUE)
        log_timed_out_request(chat_request, "Client cancelled while waiting in queue", elapsed, prompt_to_model)
        await _cancel_and_release(acquire_task, lock)
        return False
    except Exception:
        disconnect_task.cancel()
        await _cancel_and_release(acquire_task, lock)
        raise
    finally:
        if was_queued:
            queued_requests_count = max(0, queued_requests_count - 1)
            log_event(
                "queue_monitor.log",
                f"[Queue Monitor] Request left queue. Currently queued: {queued_requests_count}",
                console_enabled=LOG_CONSOLE_QUEUE,
            )


# ---------------------------------------------------------------------------
# Hardware Lifespan & Image Preprocessing
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vdevice, vlm_instance
    log_event("server.log", "STARTING HAILO GATEWAY PERSISTENT SERVER INIT STEP", console_enabled=LOG_CONSOLE_SERVER)
    try:
        vdevice = VDevice()
        vlm_instance = VLM(vdevice, HEF_MODEL_PATH)
        log_event("server.log", "MODEL RUNTIME CACHED TO ACCELERATOR ACCELERATION CORE", console_enabled=LOG_CONSOLE_SERVER)
        yield
    except Exception:
        traceback.print_exc()
        os._exit(1)
    finally:
        vlm_instance = None
        vdevice = None


app = FastAPI(title="Hailo-10H Frigate Gateway", lifespan=lifespan)


def load_image_from_source(source: str) -> Image.Image:
    if source.startswith("data:image"):
        _, base64_data = source.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(base64_data))).convert("RGB")

    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "Frigate-VLM-Gateway/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    if os.path.exists(source):
        return Image.open(source).convert("RGB")

    raise ValueError("Unsupported image reference format")


def force_static_336_matrix(img: Image.Image) -> np.ndarray:
    working = img.copy()
    working.thumbnail((TARGET_NPU_DIM, TARGET_NPU_DIM), Image.Resampling.LANCZOS)

    delta_w = TARGET_NPU_DIM - working.width
    delta_h = TARGET_NPU_DIM - working.height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )

    square_img = ImageOps.expand(working, padding, fill=(0, 0, 0))
    square_img = square_img.resize((TARGET_NPU_DIM, TARGET_NPU_DIM)).convert("RGB")
    return np.array(square_img, dtype=np.uint8)[:, :, ::-1]


# ---------------------------------------------------------------------------
# API Schemas & Models Endpoint
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"

    role: Optional[str] = "user"
    content: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]], List[Any]]] = ""
    index: Optional[int] = None


class ChatCompletionRequest(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"

    model: str = MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=64)
    response_format: Optional[Dict[str, Any]] = None


@app.get("/v1/models")
async def list_models():
    models_list = [
        {"id": MODEL_ID, "object": "model", "created": 1710000000, "owned_by": "hailo-edge"},
        {"id": "gpt-4o", "object": "model", "created": 1710000000, "owned_by": "hailo-edge"},
    ]
    return {"object": "list", "data": models_list}


# ---------------------------------------------------------------------------
# Message Extraction & Content Summarization
# ---------------------------------------------------------------------------
def summarize_image_reference(source: str) -> str:
    if source.startswith("data:image"):
        header, _, payload = source.partition(",")
        return f"{header},<base64 redacted; {len(payload)} chars>"
    return source[:237] + "..." if len(source) > 240 else source


def _truncate(text: str, max_len: int = 1000) -> str:
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _summarize_single_item(item: Any) -> Any:
    if isinstance(item, str):
        return {"type": "text", "length": len(item), "text": _truncate(item)}
    if isinstance(item, dict):
        if item.get("type") == "image_url" or "image_url" in item:
            url_val = item.get("image_url", {})
            url_str = url_val.get("url", "") if isinstance(url_val, dict) else str(url_val or "")
            return {"type": "image_url", "url": summarize_image_reference(url_str)}
        text_val = str(item.get("text", ""))
        return {"type": item.get("type", "text"), "length": len(text_val), "text": _truncate(text_val)}
    return item


def summarize_message_content(content: Union[str, Dict[str, Any], List[Dict[str, Any]], List[Any]]) -> Any:
    if isinstance(content, list):
        return [_summarize_single_item(item) for item in content]
    return _summarize_single_item(content)


def _iter_message_items(messages: List[Any]):
    for msg in messages:
        content = getattr(msg, "content", None) if hasattr(msg, "content") else (msg.get("content") if isinstance(msg, dict) else None)
        if content is None:
            continue
        if isinstance(content, list):
            yield from content
        else:
            yield content


def extract_message_content_text(messages: List[Any]) -> str:
    extracted_texts: List[str] = []
    for item in _iter_message_items(messages):
        if isinstance(item, str):
            text_val = item.strip()
        elif isinstance(item, dict) and (item.get("type") == "text" or "text" in item):
            text_val = str(item.get("text", "")).strip()
        else:
            continue
        if text_val:
            extracted_texts.append(text_val)
    return "\n\n".join(extracted_texts).strip()


def extract_message_images(messages: List[Any], max_images: int = 1) -> List[Image.Image]:
    collected_images: List[Image.Image] = []
    for item in _iter_message_items(messages):
        if isinstance(item, dict) and (item.get("type") == "image_url" or "image_url" in item):
            url_val = item.get("image_url")
            url_str = url_val.get("url", "") if isinstance(url_val, dict) else str(url_val or "")
            try:
                collected_images.append(load_image_from_source(url_str))
                if len(collected_images) >= max_images:
                    break
            except Exception as err:
                log_event("server.log", f"Image load failed: {err}", console_enabled=LOG_CONSOLE_SERVER)
    return collected_images


def extract_section_with_header(text: str, header_name: str) -> str:
    if not text:
        return ""
    pattern = rf"(?ims)(^\s*#+\s*{re.escape(header_name)}[^\n\r]*[\r\n]+.*?)(?=^\s*#+|\Z)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    alt_pattern = rf"(?ims)(^\s*{re.escape(header_name)}\s*:?[\r\n]+.*?)(?=^\s*(?:#+|[A-Z][a-zA-Z0-9 _]+:)|\Z)"
    alt_match = re.search(alt_pattern, text)
    if alt_match:
        return alt_match.group(1).strip()

    return ""


def condense_prompt(text: str, max_chars: int = 1000) -> str:
    if not text:
        return ""
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean

    paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]
    condensed_parts = [paragraphs[0][:500]] if paragraphs else []

    for line in clean.splitlines():
        line_clean = line.strip()
        lower = line_clean.lower()
        if any(lower.startswith(k) for k in ["camera:", "zone:", "time:", "timestamp:", "location:"]):
            if line_clean not in condensed_parts:
                condensed_parts.append(line_clean)

    result = "\n".join(condensed_parts).strip()
    if not result or len(result) < 25:
        result = "Analyze the security camera image and describe the observed activities, people, and vehicles."

    return result[:max_chars]


def build_model_prompt_log(message_content_text: str, cleaned_prompt: str = "") -> str:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"--- LOGGING PROMPT TO MODEL [{timestamp}] ---"]
    seq_section = (
        extract_section_with_header(message_content_text, "Sequence Details")
        or extract_section_with_header(message_content_text, "Sequence")
    )
    if seq_section:
        lines.append(f"{seq_section}\n")

    objects_section = (
        extract_section_with_header(message_content_text, "Objects in Scene")
        or extract_section_with_header(message_content_text, "Objects")
        or extract_section_with_header(message_content_text, "Tracked Objects")
    )
    if objects_section:
        lines.append(f"{objects_section}\n")

    if not seq_section and not objects_section and cleaned_prompt:
        lines.append(f"prompt: {cleaned_prompt}\n")

    lines.append("-------------------------------")
    return "\n".join(lines)


def log_incoming_frigate_request(request: ChatCompletionRequest, force: bool = False) -> None:
    request_summary = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "response_format_present": request.response_format is not None,
        "response_format": request.response_format,
        "message_count": len(request.messages),
        "messages": [
            {
                "index": index,
                "role": message.role,
                "content": summarize_message_content(message.content),
            }
            for index, message in enumerate(request.messages)
        ],
    }

    content = (
        f"\n--- LOGGING INCOMING REQUEST FROM FRIGATE [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n"
        + json.dumps(request_summary, indent=2)
        + "\n---------------------------------------------\n"
    )
    log_event("incoming_requests.log", content, console_enabled=(LOG_CONSOLE_INCOMING_REQUEST or force))


# ---------------------------------------------------------------------------
# Text Normalization & English Language Cleaning
# ---------------------------------------------------------------------------
def remove_odd_characters(text: str) -> str:
    if not text:
        return ""
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    return re.sub(r"[^\x20-\x7E\r\n\t]", " ", text)


def decode_json_string_value(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value


def _format_sentences(sentences: List[str]) -> str:
    formatted = []
    for s in sentences:
        s = s.strip()
        if not s.endswith((".", "!", "?")):
            s = s + "."
        formatted.append(s[0].upper() + s[1:] if len(s) > 1 else s.upper())
    return " ".join(formatted)


def synthesize_scene_from_payload(payload: Any) -> str:
    if isinstance(payload, str):
        cleaned = payload.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"<\|.*?\|>", "", cleaned).strip()
        if "{" in cleaned or "[" in cleaned:
            try:
                parsed = json.loads(cleaned)
                synth = synthesize_scene_from_payload(parsed)
                if synth:
                    return synth
            except json.JSONDecodeError:
                pass
            key_values = re.findall(
                r'"(?:activity|description|event|summary|text|action|details|caption|scene)"\s*:\s*"((?:\\.|[^"\\])*)"',
                cleaned,
                flags=re.IGNORECASE,
            )
            if key_values:
                clean_stmts = []
                seen_norm = set()
                for kv in key_values:
                    kv_clean = clean_text_value(decode_json_string_value(kv))
                    norm = kv_clean.lower().strip()
                    if norm and norm not in seen_norm and len(norm) > 3:
                        seen_norm.add(norm)
                        clean_stmts.append(kv_clean)
                if clean_stmts:
                    return _format_sentences(clean_stmts)
        return ""

    extracted_sentences: List[str] = []

    def recurse(node: Any):
        if isinstance(node, dict):
            for key in ["description", "activity", "event", "summary", "text", "action", "details", "caption", "scene"]:
                if key in node and isinstance(node[key], str) and node[key].strip():
                    val = str(node[key]).strip()
                    if val and val not in extracted_sentences:
                        extracted_sentences.append(val)
            for k, v in node.items():
                if str(k).lower() not in ["time", "frame", "timestamp", "index", "id", "confidence", "threat"]:
                    recurse(v)
        elif isinstance(node, list):
            for item in node:
                recurse(item)
        elif isinstance(node, str) and node.strip():
            val = node.strip()
            if val and val not in extracted_sentences:
                extracted_sentences.append(val)

    recurse(payload)

    clean_sentences = []
    seen = set()
    for stmt in extracted_sentences:
        clean_stmt = clean_text_value(stmt)
        norm = clean_stmt.lower().strip()
        if norm not in seen and len(norm) > 3:
            seen.add(norm)
            clean_sentences.append(clean_stmt)

    return _format_sentences(clean_sentences) if clean_sentences else ""


def clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return synthesize_scene_from_payload(value) or ""

    text = str(value or "")
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE)
    text = text.replace("\ufffd", "")
    text = remove_odd_characters(text)

    # Strip model instruction / reasoning bleed
    text = re.sub(r"(?i)confidence\s*等级?\s*[:=]\s*\++.*", "", text)
    text = re.sub(r"(?i)daytime/on purpose\s*→.*", "", text)
    text = re.sub(r"(?i)Level\s*\d+\s*:\s*Vehicle checking the garage.*", "", text)
    text = re.sub(r"</?[a-zA-Z0-9_-]+[^>]*>", " ", text)
    text = re.sub(r"\b[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    if "{" in text or "[" in text or '"activity":' in text or '"observations":' in text or '"scene":' in text:
        synth = synthesize_scene_from_payload(text)
        if synth:
            return synth
        text = re.sub(r'^[{\[\s"]+', "", text)
        text = re.sub(r'[}\]\s",]+$', "", text)
        text = re.sub(r'"[a-zA-Z0-9_-]+"\s*:\s*', "", text)
        text = text.replace('\\"', '"').replace("{", "").replace("}", "").replace("[", "").replace("]", "")

    text = re.sub(r"\b\d+[\.\)]\s*(?:[,.;:\s]+(?=\b\d+[\.\)]|\Z))+", " ", text)

    raw_parts = re.split(r'(?:\r?\n|(?<=[.!?])\s+|(?<=\s)\d+[\.\)]\s+)', text)
    valid_sentences = []
    seen = set()
    for part in raw_parts:
        clean_part = re.sub(r'^\s*(?:\d+[\.\)]|[-*•])\s*', '', part.strip()).strip(" ,;:. '\"")
        if not clean_part:
            continue
        words = re.findall(r'[a-zA-Z]{2,}', clean_part)
        if len(words) < 2 or any(len(w) > 22 for w in words):
            continue
        if any(marker in clean_part.lower() for marker in [
            "observations:", "observades:", "messenger", "cepturity", "oko ments",
            "viewdidloads", "dexterventus", "sinoresentent", "sinoenth", "ucthth",
            "aphwnd", "stirz1", "altxol"
        ]):
            continue
        norm = clean_part.lower()
        if norm not in seen:
            seen.add(norm)
            sentence = clean_part[0].upper() + clean_part[1:]
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            valid_sentences.append(sentence)

    if valid_sentences:
        return " ".join(valid_sentences)

    return re.sub(r"\s+", " ", text).strip(" '\",;:").strip()


def ensure_clean_english_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return synthesize_scene_from_payload(value)
    text = clean_text_value(value)
    if not text:
        return ""
    if any(c in text for c in ["{", "}", "[", "]", "\\", '":']):
        synth = synthesize_scene_from_payload(text)
        if synth:
            text = synth
        else:
            text = re.sub(r'[{}\[\]\\]', ' ', text)
            text = re.sub(r'"[a-zA-Z0-9_-]+"\s*:\s*', ' ', text)
            text = re.sub(r'"', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# JSON & Field Extraction Helpers
# ---------------------------------------------------------------------------
def extract_observation_records(source: Any, raw_text: str = "") -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen_activities = set()

    def add_record(activity: str, time_val: str = ""):
        act = ensure_clean_english_text(activity)
        if not act or len(act) < 3 or any(c in act for c in ["{", "}", "[", "]"]):
            return
        norm = act.lower()
        if norm in seen_activities:
            return
        seen_activities.add(norm)
        record = {"activity": act}
        time_clean = ensure_clean_english_text(time_val) if time_val else ""
        if time_clean and not any(c in time_clean for c in ["{", "}", "[", "]", ":+"]):
            record["time"] = time_clean
        records.append(record)

    if isinstance(source, dict):
        obs_val = source.get("observations")
        if isinstance(obs_val, list):
            for item in obs_val:
                if isinstance(item, dict):
                    add_record(item.get("activity", item.get("description", "")), item.get("time", ""))
                elif isinstance(item, str):
                    clean_item = item.strip()
                    if "{" in clean_item or '"activity":' in clean_item:
                        for act in re.findall(r'"activity"\s*:\s*"((?:\\.|[^"\\])*)"', clean_item):
                            add_record(decode_json_string_value(act))
                    else:
                        add_record(clean_item)

    candidate_text = raw_text or (source if isinstance(source, str) else "")
    if candidate_text:
        obj_matches = list(re.finditer(
            r'\{\s*"activity"\s*:\s*"((?:\\.|[^"\\])*)"(?:\s*,\s*"time"\s*:\s*"((?:\\.|[^"\\])*)")?\s*\}',
            candidate_text,
            flags=re.IGNORECASE,
        ))
        for m in obj_matches:
            add_record(decode_json_string_value(m.group(1)), decode_json_string_value(m.group(2)) if m.group(2) else "")

        if not obj_matches or len(records) < 2:
            for act in re.findall(r'"activity"\s*:\s*"((?:\\.|[^"\\])*)"', candidate_text, flags=re.IGNORECASE):
                add_record(decode_json_string_value(act))

        if not records:
            for act in re.findall(r'"(?:description|event|action)"\s*:\s*"((?:\\.|[^"\\])*)"', candidate_text, flags=re.IGNORECASE):
                add_record(decode_json_string_value(act))

        if not records:
            for part in re.split(r'(?:\r?\n|(?<=[.!?])\s+|\b\d+[\.\)]\s+)', candidate_text):
                clean_part = clean_text_value(part)
                clean_part = re.sub(r'^\s*(?:\d+[\.\)]|[-*•])\s*', '', clean_part).strip()
                words = re.findall(r'[a-zA-Z]{2,}', clean_part)
                if len(words) >= 3 and len(clean_part) >= 10:
                    add_record(clean_part)

    return records


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            parsed = json.loads(cleaned[first_brace : last_brace + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    if first_brace != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[first_brace:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        prefix = cleaned[first_brace:]
        if prefix.count('"') % 2 != 0:
            prefix += '"'
        open_brackets = prefix.count("[") - prefix.count("]")
        open_braces = prefix.count("{") - prefix.count("}")
        repaired = prefix + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (SyntaxError, ValueError):
            pass

    return {}


def limit_text(value: Any, max_chars: int) -> str:
    text = ensure_clean_english_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip(" .,;:") + "..."


def extract_json_like_fields(text: str) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    cleaned = clean_text_value(text)

    for key in ["title", "scene", "shortSummary", "description", "summary"]:
        match = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, re.DOTALL)
        if match:
            clean_val = clean_text_value(decode_json_string_value(match.group(1)))
            if clean_val:
                extracted[key] = clean_val

    obs_records = extract_observation_records({}, text)
    if obs_records:
        extracted["observations"] = obs_records

    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]*)?)', cleaned)
    if confidence_match:
        extracted["confidence"] = confidence_match.group(1)

    threat_match = re.search(r'"potential_threat_level"\s*:\s*([0-2])', cleaned)
    if threat_match:
        extracted["potential_threat_level"] = threat_match.group(1)

    concerns_match = re.search(r'"other_concerns"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
    if concerns_match:
        concern_values = []
        for concern in re.findall(r'"((?:\\.|[^"\\])*)"', concerns_match.group(1)):
            decoded = limit_text(decode_json_string_value(concern), 140)
            if decoded and decoded not in concern_values:
                concern_values.append(decoded)
            if len(concern_values) >= 5:
                break
        extracted["other_concerns"] = concern_values

    return extracted


def get_model_field(model_payload: Dict[str, Any], field_name: str) -> str:
    value = model_payload.get(field_name)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return synthesize_scene_from_payload(value)
    return ensure_clean_english_text(value)


def extract_labeled_fields(text: str) -> Dict[str, Any]:
    field_names = ["title", "scene", "shortSummary", "confidence", "other_concerns", "potential_threat_level"]
    field_pattern = "|".join(re.escape(name) for name in field_names)
    matches = list(
        re.finditer(
            rf"(?im)^\s*(?:[-*]\s*)?({field_pattern})\s*[:=]\s*(.*?)(?=^\s*(?:[-*]\s*)?(?:{field_pattern})\s*[:=]|\Z)",
            text.strip(),
            re.DOTALL,
        )
    )
    if not matches:
        return {}

    extracted: Dict[str, Any] = {}
    for match in matches:
        key = match.group(1)
        value = match.group(2).strip().strip('"\'')
        if key == "other_concerns":
            try:
                parsed_value = json.loads(value)
                extracted[key] = parsed_value if isinstance(parsed_value, list) else [str(parsed_value)]
            except json.JSONDecodeError:
                extracted[key] = [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
        else:
            extracted[key] = value
    return extracted


# ---------------------------------------------------------------------------
# Coercion & Derivation Rules
# ---------------------------------------------------------------------------
def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val_str = value.strip().lower()
        if val_str in ["true", "1", "yes"]:
            return True
        if val_str in ["false", "0", "no"]:
            return False
    return default


def coerce_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def coerce_threat_level(value: Any, default: int) -> int:
    try:
        return max(0, min(2, int(value)))
    except (TypeError, ValueError):
        return default


def derive_title(scene: str, obs_records: Optional[List[Dict[str, str]]] = None) -> str:
    clean_scene = ensure_clean_english_text(scene)
    lower = clean_scene.lower()

    if "vehicle" in lower and "person" in lower:
        return "Vehicle and Person Activity Detected"
    if "vehicle" in lower:
        return "Vehicle Activity Detected"
    if "person" in lower:
        return "Person Activity Detected"
    if "package" in lower or "delivery" in lower:
        return "Package Delivery Detected"
    if "animal" in lower or "dog" in lower or "cat" in lower:
        return "Animal Detected"

    if obs_records:
        first_act = obs_records[0].get("activity", "")
        clean_first_act = ensure_clean_english_text(first_act)
        if clean_first_act and len(clean_first_act.split()) <= 8 and not any(c in clean_first_act for c in ["{", "}", "[", "]", ":"]):
            return clean_first_act.strip(".!?, ").title()

    first_sentence = re.split(r"[.!?]", clean_scene)[0].strip()
    words = first_sentence.split()
    if 2 <= len(words) <= 8 and not any(c in first_sentence for c in ["{", "}", "[", "]", ":"]):
        return " ".join(words).title()

    return "Activity Detected"


def derive_summary(scene: str) -> str:
    clean_scene = ensure_clean_english_text(scene)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_scene) if s.strip()]
    if len(sentences) >= 2:
        summary = f"{sentences[0]} {sentences[1]}"
    elif len(sentences) == 1:
        summary = sentences[0]
    else:
        summary = clean_scene or "Activity detected on camera."

    return limit_text(summary, 320)


def derive_concerns(scene: str) -> List[str]:
    lower = ensure_clean_english_text(scene).lower()
    keywords = [
        "package", "delivery", "vehicle", "animal", "delivery driver",
        "flashlight", "weapon", "masked", "theft", "trespass",
    ]
    return [k for k in keywords if k in lower]


def derive_threat_level(scene: str) -> int:
    lower = ensure_clean_english_text(scene).lower()
    if any(k in lower for k in ["weapon", "climbing", "force", "breaking", "masked", "theft", "carrying away"]):
        return 2
    if any(k in lower for k in ["person", "loitering", "stranger", "unknown", "trespass"]):
        return 1
    return 0


def extract_schema_info(response_format: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not isinstance(response_format, dict):
        return None, None

    schema_obj = None
    if "json_schema" in response_format and isinstance(response_format["json_schema"], dict):
        schema_obj = response_format["json_schema"].get("schema")
    elif "schema" in response_format and isinstance(response_format["schema"], dict):
        schema_obj = response_format["schema"]

    if isinstance(schema_obj, dict):
        properties = schema_obj.get("properties")
        if isinstance(properties, dict):
            return schema_obj, properties

    return schema_obj, None


# ---------------------------------------------------------------------------
# Schema Mapping & Response Building
# ---------------------------------------------------------------------------
def _sanitize_output_value(val: Any) -> Any:
    if isinstance(val, str):
        return ensure_clean_english_text(val)
    if isinstance(val, list):
        clean_list = []
        for elem in val:
            if isinstance(elem, str):
                c = ensure_clean_english_text(elem)
                if c and not any(ch in c for ch in ["{", "}", "[", "]"]):
                    clean_list.append(c)
            elif isinstance(elem, dict):
                clean_list.append({k: ensure_clean_english_text(v) if isinstance(v, str) else v for k, v in elem.items()})
            else:
                clean_list.append(elem)
        return clean_list
    return val


def map_model_response_to_schema(clean_text: str, response_format: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw_chip_narrative = str(clean_text).strip()
    model_payload = (
        extract_json_object(raw_chip_narrative)
        or extract_json_like_fields(raw_chip_narrative)
        or extract_labeled_fields(raw_chip_narrative)
    )

    obs_records = extract_observation_records(model_payload, raw_chip_narrative)

    scene = ""
    for candidate_key in ["scene", "description", "summary", "content"]:
        if candidate_key in model_payload:
            scene = get_model_field(model_payload, candidate_key)
            if scene:
                break
    if not scene and obs_records:
        scene = synthesize_scene_from_payload([r["activity"] for r in obs_records])
    if not scene and model_payload:
        scene = synthesize_scene_from_payload(model_payload)
    if not scene:
        scene = ensure_clean_english_text(raw_chip_narrative)

    if not scene or any(c in scene for c in ["{", "}", "[", "]"]):
        if obs_records:
            scene = synthesize_scene_from_payload([r["activity"] for r in obs_records])
        else:
            scene = "Activity detected on security camera."

    scene = ensure_clean_english_text(scene) or "Activity detected on security camera."

    schema_obj, properties = extract_schema_info(response_format)
    target_properties = properties or DEFAULT_REVIEW_PROPERTIES
    payload_lower_map = {str(k).lower(): v for k, v in model_payload.items()}

    result: Dict[str, Any] = {}
    for prop_name, prop_spec in target_properties.items():
        if not isinstance(prop_spec, dict):
            prop_spec = {}
        prop_type = str(prop_spec.get("type", "string")).lower()
        prop_name_lower = prop_name.lower()

        raw_val = model_payload.get(prop_name)
        if raw_val is None:
            raw_val = payload_lower_map.get(prop_name_lower)

        if prop_name_lower in ["observations", "observation"]:
            items_spec = prop_spec.get("items", {})
            is_obj_item = isinstance(items_spec, dict) and items_spec.get("type") == "object"
            if is_obj_item:
                clean_obs = []
                for r in obs_records:
                    clean_act = ensure_clean_english_text(r.get("activity", ""))
                    if clean_act and not any(c in clean_act for c in ["{", "}", "[", "]"]):
                        clean_item = {"activity": clean_act}
                        clean_t = ensure_clean_english_text(r.get("time", ""))
                        if clean_t and not any(c in clean_t for c in ["{", "}", "[", "]", ":+"]):
                            clean_item["time"] = clean_t
                        clean_obs.append(clean_item)
                if not clean_obs and scene:
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", scene) if len(s.split()) >= 3]
                    clean_obs = [{"activity": s} for s in sentences[:5]]
                result[prop_name] = clean_obs[:10]
            else:
                clean_obs = []
                for r in obs_records:
                    clean_act = ensure_clean_english_text(r.get("activity", ""))
                    if clean_act and not any(c in clean_act for c in ["{", "}", "[", "]"]) and clean_act not in clean_obs:
                        clean_obs.append(clean_act)
                if not clean_obs and scene:
                    clean_obs = [s.strip() for s in re.split(r"(?<=[.!?])\s+", scene) if len(s.split()) >= 3][:5]
                result[prop_name] = clean_obs[:10]
            continue

        if prop_name_lower in ["scene", "description", "content"]:
            if raw_val is None or not str(raw_val).strip() or any(c in str(raw_val) for c in ["{", "}", "[", "]"]):
                raw_val = scene
            else:
                raw_val = ensure_clean_english_text(raw_val)

        elif prop_name_lower == "title":
            if raw_val is None or not str(raw_val).strip() or any(c in str(raw_val) for c in ["{", "}", "[", "]", ":"]):
                raw_val = derive_title(scene, obs_records)
            else:
                raw_val = ensure_clean_english_text(raw_val)

        elif prop_name_lower in ["shortsummary", "summary", "short_summary"]:
            if raw_val is None or not str(raw_val).strip() or any(c in str(raw_val) for c in ["{", "}", "[", "]", "\\"]):
                raw_val = derive_summary(scene)
            else:
                raw_val = ensure_clean_english_text(raw_val)

        elif prop_name_lower in ["other_concerns", "concerns"]:
            if raw_val is None or not str(raw_val).strip() or any(c in str(raw_val) for c in ["{", "}", "[", "]"]):
                raw_val = derive_concerns(scene)

        elif prop_name_lower in ["potential_threat_level", "threat_level"]:
            if raw_val is None or not str(raw_val).strip():
                raw_val = derive_threat_level(scene)

        elif prop_name_lower == "confidence":
            if raw_val is None or not str(raw_val).strip():
                raw_val = 0.95

        if prop_type in ["string", "str"]:
            max_len = 1500 if prop_name_lower == "scene" else (320 if "summary" in prop_name_lower else 120)
            clean_str = ensure_clean_english_text(raw_val) if raw_val is not None else ""
            result[prop_name] = limit_text(clean_str, max_len)

        elif prop_type in ["number", "float"]:
            default_num = 0.95 if "confidence" in prop_name_lower else 0.0
            result[prop_name] = coerce_float(raw_val, default=default_num)

        elif prop_type in ["integer", "int"]:
            default_int = derive_threat_level(scene) if "threat" in prop_name_lower else 0
            result[prop_name] = coerce_threat_level(raw_val, default=default_int) if "threat" in prop_name_lower else coerce_int(raw_val, default=0)

        elif prop_type in ["boolean", "bool"]:
            result[prop_name] = coerce_bool(raw_val, default=False)

        elif prop_type in ["array", "list"]:
            items_spec = prop_spec.get("items", {})
            is_obj_item = isinstance(items_spec, dict) and items_spec.get("type") == "object"

            if isinstance(raw_val, list):
                items = []
                for item in raw_val:
                    if is_obj_item and isinstance(item, dict):
                        items.append({k: ensure_clean_english_text(v) if isinstance(v, str) else v for k, v in item.items()})
                    elif isinstance(item, dict):
                        item_text = synthesize_scene_from_payload(item) or ensure_clean_english_text(item)
                        if item_text and not any(c in item_text for c in ["{", "}", "[", "]"]):
                            items.append(limit_text(item_text, 140))
                    elif item is not None:
                        item_str = ensure_clean_english_text(str(item))
                        if item_str and not any(c in item_str for c in ["{", "}", "[", "]"]):
                            items.append(limit_text(item_str, 140))
            elif isinstance(raw_val, str) and raw_val.strip():
                clean_item_str = ensure_clean_english_text(raw_val)
                items = [limit_text(clean_item_str, 140)] if clean_item_str and not any(c in clean_item_str for c in ["{", "}", "[", "]"]) else []
            else:
                items = derive_concerns(scene) if "concern" in prop_name_lower else []

            if items and isinstance(items[0], str):
                deduped, seen = [], set()
                for it in items:
                    clean_it = ensure_clean_english_text(it)
                    if clean_it and not any(c in clean_it for c in ["{", "}", "[", "]"]):
                        norm = clean_it.lower()
                        if norm not in seen:
                            seen.add(norm)
                            deduped.append(clean_it)
                items = deduped

            result[prop_name] = items[:10]

        elif prop_type == "object":
            result[prop_name] = raw_val if isinstance(raw_val, dict) else {}

        else:
            result[prop_name] = ensure_clean_english_text(raw_val) if raw_val is not None else ""

    return {k: _sanitize_output_value(v) for k, v in result.items()}


def build_openai_response(
    model_name: str,
    clean_text: str,
    response_format: Optional[Dict[str, Any]] = None,
    duration: Optional[float] = None,
    queue_time: Optional[float] = None,
    inference_time: Optional[float] = None,
    image_prep_time: Optional[float] = None,
    is_chat_mode: bool = False,
) -> Dict[str, Any]:
    structured_response = response_format is not None
    if structured_response:
        formatted_payload = map_model_response_to_schema(clean_text, response_format)
        frigate_content = json.dumps(formatted_payload)
    elif is_chat_mode:
        frigate_content = re.sub(r"<\|.*?\|>", "", clean_text).strip()
    else:
        frigate_payload_dictionary = map_model_response_to_schema(clean_text, None)
        frigate_content = frigate_payload_dictionary.get("scene", clean_text) if isinstance(frigate_payload_dictionary, dict) else clean_text
        frigate_content = ensure_clean_english_text(frigate_content)

    msg_obj = {"role": "assistant", "content": frigate_content}
    choice_item = {"index": 0, "message": msg_obj, "finish_reason": "stop"}
    usage_obj = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    response_payload = {
        "id": "chatcmpl-hailo10h",
        "object": "chat.completion",
        "created": 1710000000,
        "model": str(model_name or MODEL_ID),
        "choices": [choice_item],
        "usage": usage_obj,
    }

    if structured_response:
        mode_str = "review JSON"
    elif is_chat_mode:
        mode_str = "interactive chat"
    else:
        mode_str = "object description text"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header_suffix = f" (Duration: {duration:.2f}s)" if duration is not None else ""
    response_log_parts = [
        f"\n--- LOGGING RESPONSE SENT TO FRIGATE [{timestamp}]{header_suffix} ---",
        f"response_mode: {mode_str}",
    ]
    if structured_response:
        response_log_parts.append("mapped_values:")
        response_log_parts.append(json.dumps(formatted_payload, indent=2))
    response_log_parts.append("message_content:")
    response_log_parts.append(frigate_content)
    response_log_parts.append("full_openai_compatible_response:")
    response_log_parts.append(json.dumps(response_payload, indent=2))
    response_log_parts.append("----------------------------------------\n")
    log_event("frigate_responses.log", "\n".join(response_log_parts), console_enabled=LOG_CONSOLE_RESPONSES)

    timing_lines = [
        "=" * 60,
        f"⏱️  REQUEST TIMING SUMMARY [{timestamp}]",
        "=" * 60,
    ]
    if image_prep_time is not None:
        timing_lines.append(f"  • Image Prep Time:     {image_prep_time:.2f}s")
    if queue_time is not None:
        timing_lines.append(f"  • Queue Wait Time:     {queue_time:.2f}s")
    if inference_time is not None:
        timing_lines.append(f"  • NPU Inference Time:  {inference_time:.2f}s")
    timing_lines.append("  " + "-" * 40)
    dur_val = f"{duration:.2f}s" if duration is not None else "N/A"
    timing_lines.append(f"  ★ TOTAL DURATION:      {dur_val}  ({mode_str})")
    timing_lines.append("=" * 60 + "\n")

    log_event("request_timing.log", "\n".join(timing_lines), console_enabled=LOG_CONSOLE_TIMING)

    return response_payload


def extract_vlm_text(raw_response: Any) -> str:
    extracted = ""
    if hasattr(raw_response, "text"):
        extracted = str(raw_response.text).strip()
    elif hasattr(raw_response, "choices") and raw_response.choices:
        try:
            first_choice = raw_response.choices[0]
            if hasattr(first_choice, "text"):
                extracted = str(first_choice.text).strip()
            elif hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                extracted = str(first_choice.message.content).strip()
        except Exception:
            pass

    if not extracted:
        try:
            extracted = "".join([str(chunk) for chunk in raw_response]).strip()
        except Exception:
            pass

    if not extracted:
        for candidate in re.findall(r"['\"](.*?)['\"]", str(raw_response)):
            if len(candidate) > 5 and "LLMGenerator" not in candidate and "object at" not in candidate:
                extracted = candidate.strip()
                break

    if extracted:
        extracted = re.sub(r"<\|.*?\|>", "", extracted).strip()

    return extracted


async def _reset_vlm_context(vlm: Any, loop: asyncio.AbstractEventLoop, log: bool = False) -> None:
    for method_name, msg in [
        ("clear_context", "Cache Monitor: Native context cache wiped successfully."),
        ("reset_context", "Cache Monitor: Native context cache reset successfully."),
        ("clear_history", "Cache Monitor: History attention log wiped successfully."),
    ]:
        method = getattr(vlm, method_name, None)
        if callable(method):
            try:
                await loop.run_in_executor(None, method)
                if log:
                    log_event("server.log", msg, console_enabled=LOG_CONSOLE_SERVER)
                return
            except Exception:
                pass


# ---------------------------------------------------------------------------
# OpenAI Chat Completions Endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    global vlm_instance
    request_start_time = time.time()

    if not vlm_instance:
        raise HTTPException(status_code=503, detail="NPU hardware uninitialized.")

    t_prep_start = time.time()
    message_content_text = extract_message_content_text(request.messages)
    collected_images = extract_message_images(request.messages, max_images=1)

    has_images = bool(collected_images)
    if has_images:
        primary_image_handle = collected_images[0]
        bgr_matrix_frame = force_static_336_matrix(primary_image_handle)
    else:
        bgr_matrix_frame = np.zeros((TARGET_NPU_DIM, TARGET_NPU_DIM, 3), dtype=np.uint8)

    image_prep_time = time.time() - t_prep_start

    structured_response = request.response_format is not None
    is_chat_mode = not has_images and not structured_response

    schema_obj, properties = extract_schema_info(request.response_format)

    if structured_response:
        condensed_text = condense_prompt(message_content_text, max_chars=600)
        if not condensed_text:
            condensed_text = "Analyze what is happening in this security camera frame."
        keys = (
            list(properties.keys())
            if properties
            else (list(schema_obj.get("properties", {}).keys()) if isinstance(schema_obj, dict) and isinstance(schema_obj.get("properties"), dict) else [])
        )
        keys_str = ", ".join(keys) if keys else "observations, scene, title, shortSummary"
        cleaned_prompt = (
            f"{condensed_text}\n\n"
            f"Return only a valid JSON object with keys: {keys_str}. "
            f"Describe the observed activities, objects, and people. "
            f"Do not include markdown fences, comments, or explanatory text outside the JSON object."
        )
    elif is_chat_mode:
        chat_turns = []
        for msg in request.messages:
            role = getattr(msg, "role", "user") or "user"
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                text_items = [str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content]
                text_str = " ".join(t.strip() for t in text_items if t.strip())
            elif isinstance(content, dict):
                text_str = str(content.get("text", "")).strip()
            else:
                text_str = str(content or "").strip()

            if not text_str:
                continue

            if role.lower() == "system":
                continue
            chat_turns.append(f"{role.capitalize()}: {text_str}")

        conversation_str = "\n".join(chat_turns).strip()
        if not conversation_str:
            conversation_str = f"User: {message_content_text.strip() or 'Hello!'}"

        cleaned_prompt = (
            "You are a helpful assistant for Frigate NVR. "
            "Answer the user's question helpfully, politely, and concisely.\n\n"
            f"{conversation_str}\n\nAssistant:"
        )
    else:
        condensed_text = condense_prompt(message_content_text, max_chars=600)
        if not condensed_text:
            condensed_text = "Analyze what is happening in this security camera frame."
        cleaned_prompt = (
            f"{condensed_text}\n\n"
            f"Return only a concise plain-language description of what you see in this security frame."
        )

    structured_prompt_list = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": cleaned_prompt},
            ],
        }
    ]

    prompt_to_model = build_model_prompt_log(message_content_text, cleaned_prompt)

    queue_start_time = time.time()
    if not await acquire_lock_or_abort(
        npu_lock,
        http_request,
        chat_request=request,
        prompt_to_model=prompt_to_model,
    ):
        raise HTTPException(status_code=499, detail="Client Closed Request")
    queue_time = time.time() - queue_start_time

    loop = asyncio.get_running_loop()
    try:
        log_incoming_frigate_request(request, force=is_chat_mode)
        req_mode_label = "review JSON" if structured_response else ("interactive chat" if is_chat_mode else "object description text")
        log_event("server.log", f"Frigate request mode: {req_mode_label}", console_enabled=LOG_CONSOLE_SERVER)

        log_event("model_prompts.log", f"\n{prompt_to_model}\n", console_enabled=LOG_CONSOLE_PROMPTS)

        await _reset_vlm_context(vlm_instance, loop, log=True)

        def _run_vlm_inference() -> str:
            raw = vlm_instance.generate(structured_prompt_list, [bgr_matrix_frame])
            return extract_vlm_text(raw)

        inference_start_time = time.time()
        clean_output = await loop.run_in_executor(None, _run_vlm_inference)
        inference_time = time.time() - inference_start_time

        if not clean_output:
            clean_output = "No narrative could be parsed from the Hailo VLM core response instance."

        model_raw_log = f"\n--- LOGGING RESPONSE OF THE MODEL [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n{clean_output}\n--------------------------------------\n"
        log_event("model_raw_responses.log", model_raw_log, console_enabled=LOG_CONSOLE_MODEL_RAW)

        duration = time.time() - request_start_time
        return build_openai_response(
            request.model,
            clean_output,
            request.response_format,
            duration=duration,
            queue_time=queue_time,
            inference_time=inference_time,
            image_prep_time=image_prep_time,
            is_chat_mode=is_chat_mode,
        )

    except Exception as err:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hailo VLM execution engine failure: {err}")
    finally:
        await _reset_vlm_context(vlm_instance, loop, log=False)
        npu_lock.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
