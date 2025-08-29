import os
import io
import json
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from PIL import Image

from transformers import pipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from google.cloud import vision

# ──────────────────────────────────────────────────────────
# 로깅 & 앱 
# ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("d-tect-model")
app = FastAPI()


# ──────────────────────────────────────────────────────────
# 경로/환경 기본값 (상대경로 기반)
# ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.resolve()

# 스프링 콜백 URL
# 해당 주소는 학원 + 제 맥북 IP. 환경에 따라 수정 필요!
SPRING_CALLBACK_URL = os.getenv("SPRING_CALLBACK_URL", "http://192.168.219.51:8081/api/analysis/callback")

# 로그 저장 루트 (기본: ./logs)
# 확인이 필요하나 채팅 로그가 제대로 저장이 안되는거같음
# 확인해볼 예정
LOG_ROOT: Path = Path(os.environ.get("DTECT_LOG_DIR", PROJECT_ROOT / "logs")).resolve()
DATA_DIR: Path = LOG_ROOT / "data"   # chat_log.json
JSON_DIR: Path = LOG_ROOT / "json"   # classified_chats.json
DATA_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

CHAT_LOG_PATH: Path = DATA_DIR / "chat_log.json"
CLASSIFIED_LOG_PATH: Path = JSON_DIR / "classified_chats.json"

# 안전모드 스위치 (없으면 '0')
# 에러 검증용인데 잘 작동하는지 모르겠음
DISABLE_VISION = os.getenv("DTECT_DISABLE_VISION", "0") == "1"
DISABLE_LLM    = os.getenv("DTECT_DISABLE_LLM", "0") == "1"
ENABLE_LOGS    = os.getenv("DTECT_ENABLE_LOGS", "1") == "1"  # 로그 JSON 저장 on/off

# ──────────────────────────────────────────────────────────
# Google Vision 자격증명 자동 탐지 (상대경로 우선)
# ──────────────────────────────────────────────────────────
def _ensure_vision_credentials():
    # 1) 이미 환경변수에 유효 경로가 있으면 그대로 사용
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        log.info("Using GOOGLE_APPLICATION_CREDENTIALS from env: %s", env_path)
        return

    # 2) 사용자 지정 오버라이드 환경변수(DTECT_VISION_KEY_PATH)
    custom = os.getenv("DTECT_VISION_KEY_PATH")
    if custom and Path(custom).exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = custom
        log.info("Using DTECT_VISION_KEY_PATH: %s", custom)
        return

    # 3) ./credentials/*.json 중 첫번째 파일을 자동 선택
    cred_dir = PROJECT_ROOT / "credentials"
    if cred_dir.is_dir():
        json_files = sorted(cred_dir.glob("*.json"))
        if json_files:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(json_files[0])
            log.info("Using credentials (relative): %s", json_files[0])
            return

    # 4) 없으면 비활성로 둠(안전모드로 fallback)
    log.warning("No Vision credential found. Set GOOGLE_APPLICATION_CREDENTIALS or put a .json under ./credentials/")

_ensure_vision_credentials()

# ──────────────────────────────────────────────────────────
# 분류 라벨/가이드
# ──────────────────────────────────────────────────────────
ALLOWED = [
    "VIOLENCE", "DEFAMATION", "STALKING", "SEXUAL",
    "LEAK", "BULLYING", "CHANTAGE", "EXTORTION"
]

GUIDE = """
• 한 줄은 여러 라벨 동시 가능.
• SEXUAL: 성적 의도/행위의 암시·요구·대상화 등.
• VIOLENCE: 물리적 강제·폭력·협박·위협 등.
• BULLYING: 집단 배제/무시/왕따·강퇴/차단 유도·비꼼 등.
• DEFAMATION: 사실/허위사실 유포로 평판 손상.
• LEAK: 신상·개인정보 공개·유포.
• STALKING: 지속 추적·감시·집요한 연락.
• CHANTAGE: 비밀을 빌미로 순응 강요.
• EXTORTION: 폭력·협박으로 금품/이익 강요.
"""

JSON_SCHEMA = """[
  {"label":"VIOLENCE","count":0},
  {"label":"DEFAMATION","count":0},
  {"label":"STALKING","count":0},
  {"label":"SEXUAL","count":0},
  {"label":"LEAK","count":0},
  {"label":"BULLYING","count":0},
  {"label":"CHANTAGE","count":0},
  {"label":"EXTORTION","count":0}
]"""

# ──────────────────────────────────────────────────────────
# 지연 로딩(서버 기동 실패 방지)
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
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set → LLM disabled.")
        return None
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=600,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "이 시스템은 욕설로 분류된 텍스트의 사이버폭력 유형을 구분합니다.\n"
             f"라벨 후보: {', '.join(ALLOWED)}\n"
             "오직 JSON만 출력.\n\n응답 형식:\n" + JSON_SCHEMA
            ),
            ("human", "{text}")
        ])
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
    parts = line.split(' ', 1)
    return {"user": parts[0], "text": parts[1]} if len(parts) == 2 else {"user": "Unknown", "text": parts[0]}

def pick_one(llm_list) -> Dict[str, Any]:
    try:
        if isinstance(llm_list, list) and llm_list:
            pos = [d for d in llm_list if int(d.get("count", 0)) > 0]
            chosen = max(pos, key=lambda d: int(d["count"])) if pos else llm_list[0]
            return {"label": str(chosen["label"]), "count": int(chosen.get("count", 0))}
        if isinstance(llm_list, dict) and "label" in llm_list:
            return {"label": str(llm_list["label"]), "count": int(llm_list.get("count", 0))}
    except Exception:
        pass
    return {"label": "UNKNOWN", "count": 0}

def save_json(data: Any, path: Path):
    if not ENABLE_LOGS:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning("save_json failed (%s): %s", path, e)

def ocr_lines_from_image_bytes(png_bytes: bytes) -> List[str]:
    """이미지 바이트 → (아래쪽 비율 크롭) → Vision OCR → 라인 리스트"""
    cli = get_vision_client()
    if cli is None:
        # 안전모드: OCR 비활성 → 빈 라인
        return []

    # 하단 40% 영역 크롭
    with Image.open(io.BytesIO(png_bytes)) as im:
        w, h = im.size
        y_min = int(h * 0.6)
        cropped = im.crop((0, y_min, w, h))
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

def classify_lines(chat_lines: List[str]) -> List[Dict[str, Any]]:
    """OCR 라인들 → 불링 의심만 추려 ModelMessage 리스트로 반환"""
    out: List[Dict[str, Any]] = []
    clf = get_unsmile()
    chain = get_llm_chain()  # None일 수 있음(안전모드/키 없음)

    # 로그용: 원시 라인 저장 (user/text 분리 형태로)
    parsed_for_log = [split_user_text(line) for line in chat_lines]
    save_json(parsed_for_log, CHAT_LOG_PATH)

    for line in chat_lines:
        st = split_user_text(line)
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
            one = {"label": "UNKNOWN", "count": 0}
            if chain is not None:
                try:
                    res = chain.invoke({"text": text})
                    llm_data = json.loads(res.content)
                    one = pick_one(llm_data)
                except Exception as e:
                    log.warning("LLM classify fail: %s", e)

            out.append({
                "user": st["user"],
                "text": text,
                "score": f"{score:.2f}",   # 문자열 점수
                "classification": one      # 단일 객체
            })

    save_json(out, CLASSIFIED_LOG_PATH)
    return out

def post_callback(url: str, payload: List[Dict[str, Any]]) -> None:
    if not url:
        return
    try:
        import requests
        r = requests.post(url, json=payload, timeout=(10, 30))
        log.info("[callback] POST %s → %s", url, r.status_code)
        if r.status_code >= 400:
            log.warning("[callback body] %s", r.text[:300])
    except Exception as e:
        log.warning("callback error: %s", e)

# ──────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    deps = {
        "vision_enabled": get_vision_client() is not None,
        "llm_enabled": get_llm_chain() is not None,
        "log_dir": str(LOG_ROOT)
    }
    return {"status": "ok", "message": "pong", "deps": deps}

@app.post("/predict")
async def predict(file: UploadFile = File(...), background: BackgroundTasks = None):
    """
    업로드된 스크린샷 1장에 대해:
    - 하단 영역 크롭 → OCR → 라인 분리 → UNSMILE(+LLM) 분류
    - 분류 결과 리스트 반환
    - 동시에 SPRING_CALLBACK_URL 로 결과도 전송(백그라운드)
    - 남아서 이야기 했듯이! sid 라는 값으로 인해 현재 콜백은 작동이 안됨
    """
    try:
        content = await file.read()
        lines = ocr_lines_from_image_bytes(content)
        results = classify_lines(lines) if lines else []

        if background is not None and SPRING_CALLBACK_URL:
            background.add_task(post_callback, SPRING_CALLBACK_URL, results)

        return results
    except Exception as e:
        log.exception("predict error: %s", e)
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
