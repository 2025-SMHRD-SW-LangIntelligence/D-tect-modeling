import os
import io
import json
import re
import time
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import difflib

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from PIL import Image

from transformers import pipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from google.cloud import vision

# ──────────────────────────────────────────────────────────
# 로깅 & 앱
# ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("d-tect-model")
app = FastAPI()

# ──────────────────────────────────────────────────────────
# 경로/환경 기본값
# ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.resolve()

# 스프링 콜백 URL (1: 모델 메시지 저장, 2: 라벨 카운트/등급 누적)
SPRING_CALLBACK_URL = os.getenv(
    "SPRING_CALLBACK_URL",
    "http://127.0.0.1:8081/api/analysis/callback-model"  # 필요시 8080으로 변경
)
SPRING_CALLBACK_URL_2 = os.getenv(
    "SPRING_CALLBACK_URL_2",
    "http://127.0.0.1:8081/api/analysis/callback"
)

# 로그 저장 루트 (기본: ./logs)
LOG_ROOT: Path = Path(os.environ.get("DTECT_LOG_DIR", PROJECT_ROOT / "logs")).resolve()
LOG_ROOT.mkdir(parents=True, exist_ok=True)
CHAT_LOG_PATH: Path = LOG_ROOT / "chat_log.json"
CLASSIFIED_LOG_PATH: Path = LOG_ROOT / "classified_chats.json"

# 안전모드 스위치
DISABLE_VISION = os.getenv("DTECT_DISABLE_VISION", "0") == "1"
DISABLE_LLM    = os.getenv("DTECT_DISABLE_LLM", "0") == "1"
ENABLE_LOGS    = os.getenv("DTECT_ENABLE_LOGS", "1") == "1"

# 중복 억제 (캡처 주기 중복 방지)
DEDUP_ENABLED        = os.getenv("DTECT_DEDUP_ENABLED", "1") == "1"
DEDUP_TTL_SEC        = float(os.getenv("DTECT_DEDUP_TTL_SEC", "12"))
DEDUP_SIM_THRESHOLD  = float(os.getenv("DTECT_DEDUP_SIM_THRESHOLD", "0.985"))
DEDUP_MAX_ENTRIES    = int(os.getenv("DTECT_DEDUP_MAX_ENTRIES", "400"))

_RECENT_DETECTIONS: Dict[str, deque] = {}

# ──────────────────────────────────────────────────────────
# Google Vision 자격증명 자동 탐지
# ──────────────────────────────────────────────────────────
def _ensure_vision_credentials():
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        log.info("Using GOOGLE_APPLICATION_CREDENTIALS from env: %s", env_path)
        return
    custom = os.getenv("DTECT_VISION_KEY_PATH")
    if custom and Path(custom).exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = custom
        log.info("Using DTECT_VISION_KEY_PATH: %s", custom)
        return
    cred_dir = PROJECT_ROOT / "credentials"
    if cred_dir.is_dir():
        json_files = sorted(cred_dir.glob("*.json"))
        if json_files:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(json_files[0])
            log.info("Using credentials (relative): %s", json_files[0])
            return
    log.warning("No Vision credential found. Put a .json under ./credentials/ or set GOOGLE_APPLICATION_CREDENTIALS")

_ensure_vision_credentials()

# ──────────────────────────────────────────────────────────
# OpenAI API Key 파일 로드
# ──────────────────────────────────────────────────────────
OPENAI_KEY_FILES = ["openai.key", "openai_api_key.txt", "openai.json"]

@lru_cache
def get_openai_api_key() -> Tuple[Optional[str], str]:
    k = os.getenv("OPENAI_API_KEY")
    if k:
        return k.strip(), "env:OPENAI_API_KEY"

    path_env = os.getenv("OPENAI_API_KEY_FILE") or os.getenv("DTECT_OPENAI_KEY_PATH")
    candidates = []
    if path_env:
        candidates.append(Path(path_env))

    cred_dir = PROJECT_ROOT / "credentials"
    for name in OPENAI_KEY_FILES:
        candidates.append(cred_dir / name)

    for p in candidates:
        try:
            if not p or not p.exists():
                continue
            if p.suffix.lower() in (".json", ".jsn"):
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for field in ("api_key", "openai_api_key", "OPENAI_API_KEY", "key"):
                    v = data.get(field)
                    if v:
                        return str(v).strip(), f"file:{p.name}"
            else:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            return s, f"file:{p.name}"
        except Exception as e:
            log.warning("Failed reading OpenAI key file %s: %s", p, e)
    return None, "none"

# ──────────────────────────────────────────────────────────
# 라벨 후보 (LLM 안내)
# ──────────────────────────────────────────────────────────
ALLOWED = [
    "VIOLENCE", "DEFAMATION", "STALKING", "SEXUAL",
    "LEAK", "BULLYING", "CHANTAGE", "EXTORTION"
]

def _labels_envelope_schema() -> dict:
    return {
        "name": "cyberbullying_labels",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "count"],
                        "properties": {
                            "label": {"type": "string", "enum": ALLOWED},
                            "count": {"type": "integer", "minimum": 1, "maximum": 5}
                        }
                    }
                }
            }
        }
    }

# ──────────────────────────────────────────────────────────
# 지연 로딩 (UNSMILE / LLM / Vision)
# ──────────────────────────────────────────────────────────
@lru_cache
def get_unsmile():
    log.info("Loading UNSMILE pipeline...")
    return pipeline("text-classification", model="smilegate-ai/kor_unsmile")

@lru_cache
def get_llm_chain() -> Optional[ChatPromptTemplate]:
    if DISABLE_LLM:
        log.warning("LLM disabled by env (DTECT_DISABLE_LLM=1).")
        return None

    key, src = get_openai_api_key()
    if not key:
        log.warning("OPENAI_API_KEY not found (src=%s) → LLM disabled.", src)
        return None

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=400,
            api_key=key,
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": _labels_envelope_schema()
                }
            }
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "역할: 채팅 메시지의 사이버폭력 유형을 분류하는 분류기.\n"
             "안내: 라벨 후보는 다음과 같음 → " + ", ".join(ALLOWED) + "\n"
             "출력: 오직 JSON object만. 최상위 키는 'items'이며, 각 원소는 "
             "객체로서 'label'(enum)과 'count'(1..5)을 가진다.\n"
             "예시: {{\"items\": [{{\"label\": \"BULLYING\", \"count\": 1}}]}}\n"
             "정책: 유해 표현도 분류 목적상 그대로 고려. 설명문/사과문/서론 금지. JSON 이외 금지.\n"
            ),
            ("human", "분석할 문장: \"{text}\"")
        ])
        log.info("LLM ready (key source: %s)", src)
        return prompt | llm
    except Exception as e:
        log.warning("LLM init failed: %s", e)
        return None

@lru_cache
def get_vision_client() -> Optional[vision.ImageAnnotatorClient]:
    if DISABLE_VISION:
        log.warning("Vision disabled by env (DTECT_DISABLE_VISION=1).")
        return None
    try:
        return vision.ImageAnnotatorClient()
    except Exception as e:
        log.warning("Vision init failed: %s", e)
        return None

# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────
def remove_prefix_text(text: str) -> str:
    text = re.sub(r'^(1|F|더|3|5|ㅑ|L|ㄹ|②)\s+', '', text)
    text = re.sub(r'https\s*:\s*//\s*w\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def add_space_before_question_mark(text: str) -> str:
    return re.sub(r'(\S)\?', r'\1 ?', text)

def split_user_text(line: str) -> Dict[str, str]:
    badge_pattern = r'^[\|♥①②③④⑤⑥⑦⑧⑨•●★\s]+'
    cleaned_line = re.sub(badge_pattern, '', line.strip())
    parts = cleaned_line.split(' ', 1)
    if len(parts) == 2:
        return {"user": parts[0], "text": parts[1]}
    else:
        return {"user": "Unknown", "text": parts[0]}

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_first_json(s: str) -> Optional[Any]:
    s0 = _strip_code_fences(s)
    try:
        return json.loads(s0)
    except Exception:
        pass
    try:
        i = s0.find('['); j = s0.rfind(']')
        if i != -1 and j != -1 and j > i:
            return json.loads(s0[i:j+1])
    except Exception:
        pass
    try:
        i = s0.find('{'); j = s0.rfind('}')
        if i != -1 and j != -1 and j > i:
            return json.loads(s0[i:j+1])
    except Exception:
        pass
    return None

def parse_llm_to_list(x: Any) -> List[Dict[str, Any]]:
    try:
        data = x.content if hasattr(x, "content") else x
        if isinstance(data, str):
            data = _extract_first_json(data)

        out: List[Dict[str, Any]] = []

        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            for el in data["items"]:
                if not isinstance(el, dict):
                    continue
                label = str(el.get("label", "")).upper().strip()
                try:
                    cnt = int(el.get("count", 0))
                except Exception:
                    cnt = 0
                if label and cnt > 0:
                    out.append({"label": label, "count": cnt})
            return out

        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    cnt = int(v)
                except Exception:
                    continue
                if cnt > 0:
                    out.append({"label": str(k).upper().strip(), "count": cnt})
            return out

        if isinstance(data, list):
            for el in data:
                if not isinstance(el, dict):
                    continue
                label = str(el.get("label", "")).upper().strip()
                try:
                    cnt = int(el.get("count", 0))
                except Exception:
                    cnt = 0
                if label and cnt > 0:
                    out.append({"label": label, "count": cnt})
            return out
    except Exception:
        pass
    return []

def append_json_lines(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not ENABLE_LOGS or not rows:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")
    except Exception as e:
        log.warning("append_json_lines failed (%s): %s", path, e)

def ocr_lines_from_image_bytes(
    png_bytes: bytes,
    crop_top_ratio: float = 0.15,
    crop_bottom_ratio: float = 0.1
) -> List[str]:
    cli = get_vision_client()
    if cli is None:
        return []
    with Image.open(io.BytesIO(png_bytes)) as im:
        w, h = im.size
        y_min = int(h * crop_top_ratio)
        y_max = h - int(h * crop_bottom_ratio)
        if y_min >= y_max:
            log.warning(f"Invalid crop ratios: top={crop_top_ratio}, bottom={crop_bottom_ratio}")
            return []
        cropped = im.crop((0, y_min, w, y_max))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        cropped_bytes = buf.getvalue()

    img = vision.Image(content=cropped_bytes)
    resp = cli.text_detection(image=img)
    texts = resp.text_annotations
    if not texts:
        return []
    lines = texts[0].description.split('\n')
    cleaned = []
    for line in lines:
        x = add_space_before_question_mark(remove_prefix_text(line))
        if x:
            cleaned.append(x)
    return cleaned

def _canon_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def _should_suppress_duplicate(anal_key: str, text: str) -> bool:
    if not DEDUP_ENABLED:
        return False
    now = time.time()
    canon = _canon_text(text)
    dq = _RECENT_DETECTIONS.setdefault(anal_key, deque())
    ttl_cut = now - DEDUP_TTL_SEC
    while dq and dq[0][0] < ttl_cut:
        dq.popleft()
    for _, canon_i in dq:
        if canon == canon_i:
            return True
        if difflib.SequenceMatcher(a=canon_i, b=canon).ratio() >= DEDUP_SIM_THRESHOLD:
            return True
    dq.append((now, canon))
    if len(dq) > DEDUP_MAX_ENTRIES:
        for _ in range(len(dq) - DEDUP_MAX_ENTRIES):
            dq.popleft()
    return False

def classify_lines(chat_lines: List[str], anal_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload: List[Dict[str, Any]] = []
    log_detected: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    if not chat_lines:
        return payload, log_detected, raw_rows

    clf = get_unsmile()
    chain = get_llm_chain()

    for line in chat_lines:
        st = split_user_text(line)
        raw_rows.append({"user": st["user"], "text": st["text"]})
        text = st.get("text", "")
        if not text:
            continue

        try:
            unsmile = clf(text)
            label = unsmile[0]["label"]
            score = float(unsmile[0]["score"])
        except Exception as e:
            log.exception("UNSMILE error: %s", e)
            continue

        if label != "clean" and score > 0.80:
            if _should_suppress_duplicate(anal_key, text):
                log_detected.append({
                    "user": st["user"],
                    "text": text,
                    "score": f"{score:.2f}",
                    "classification": [{"label": "BULLYING", "count": 1}],
                    "suppressed": True,
                    "reason": "duplicate_within_ttl"
                })
                continue

            cls_list: List[Dict[str, Any]] = []
            reason = None
            if chain is not None:
                try:
                    llm_res = chain.invoke({"text": text})
                    cls_list = parse_llm_to_list(llm_res)
                    if not cls_list:
                        reason = f"LLM empty/unparseable → default BULLYING"
                except Exception as e:
                    log.exception("LLM classify error. Text: '%s'", text)
                    reason = f"LLM exception({e.__class__.__name__}) → default BULLYING"

            if not cls_list:
                cls_list = [{"label": "BULLYING", "count": 1}]

            item_backend = {
                "user": st["user"],
                "text": text,
                "score": f"{score:.2f}",
                "classification": cls_list
            }
            payload.append(item_backend)

            log_item = dict(item_backend)
            if reason:
                log_item["llm_reason"] = reason
            log_detected.append(log_item)

    return payload, log_detected, raw_rows

def post_callback(callback_url: str, payload: List[Dict[str, Any]]) -> None:
    if not callback_url or not payload:
        return
    try:
        import requests
        r = requests.post(callback_url, json=payload, timeout=(10, 30))
        log.info("[callback] POST %s → %s", callback_url, r.status_code)
        if r.status_code >= 400:
            log.warning("[callback body] %s", r.text[:300])
    except Exception as e:
        log.warning("callback error: %s", e)

def post_callbacks(urls: List[str], anal_id: Optional[int], payload: List[Dict[str, Any]], background: Optional[BackgroundTasks] = None):
    if not payload or anal_id is None:
        return
    for base in urls:
        if not base:
            continue
        url = f"{base.rstrip('/')}?analId={anal_id}"
        if background is not None:
            background.add_task(post_callback, url, payload)
        else:
            post_callback(url, payload)

# ──────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    key, src = get_openai_api_key()
    deps = {
        "vision_enabled": get_vision_client() is not None,
        "llm_enabled": (not DISABLE_LLM) and bool(key),
        "llm_key_source": src,
        "log_dir": str(LOG_ROOT),
        "dedup_enabled": DEDUP_ENABLED,
        "dedup_ttl_sec": DEDUP_TTL_SEC,
        "dedup_sim_threshold": DEDUP_SIM_THRESHOLD,
    }
    return {"status": "ok", "message": "pong", "deps": deps}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    background: BackgroundTasks = None,
    analId: Optional[int] = Query(default=None),
    crop_top: float = Query(default=0.15, ge=0.0, le=1.0),
    crop_bottom: float = Query(default=0.1, ge=0.0, le=1.0),
):
    """
    - 이미지 크롭 → OCR → UNSMILE(+LLM) → 탐지된 라인 payload 생성
    - 콜백: 탐지시 /callback-model, /callback 두 군데로 동시에 전송(analId 필요)
    - 응답: { flagged: bool, labels: string[] }
    """
    try:
        content: bytes = await file.read()
        lines = ocr_lines_from_image_bytes(
            content,
            crop_top_ratio=crop_top,
            crop_bottom_ratio=crop_bottom
        )

        anal_key = f"anal:{analId}" if analId is not None else "global"
        payload_items, log_detected, raw_rows = classify_lines(lines, anal_key=anal_key)

        append_json_lines(CHAT_LOG_PATH, raw_rows)
        append_json_lines(CLASSIFIED_LOG_PATH, log_detected)

        # === 두 콜백으로 동시 전송 ===
        if payload_items and analId is not None:
            post_callbacks(
                [SPRING_CALLBACK_URL, SPRING_CALLBACK_URL_2],
                analId,
                payload_items,
                background
            )

        # 응답 요약 라벨
        unique_labels = []
        seen = set()
        for m in payload_items:
            for lc in (m.get("classification") or []):
                lbl = str(lc.get("label", "")).upper().strip()
                if lbl and lbl not in seen and lc.get("count", 0) > 0:
                    seen.add(lbl)
                    unique_labels.append(lbl)

        return {"flagged": bool(payload_items), "labels": unique_labels}
    except Exception as e:
        log.exception("predict error: %s", e)
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
