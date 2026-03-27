import os
import re
import logging
import httpx
from google import genai
from google.genai import types

from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Env vars ─────────────────────────────────────────────────────────────────
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# ── LINE SDK setup ────────────────────────────────────────────────────────────
handler = WebhookHandler(LINE_CHANNEL_SECRET)
line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# ── Gemini client ─────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Google Sheets prompt loader ───────────────────────────────────────────────
SHEET_ID = "1QjU_OVHcEEcxSpVguExAh2rmmVJBK9KLDhksIVRs4vM"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

FALLBACK_PROMPT = "你是一個事實查核助手，請用繁體中文回覆。"

def load_prompt() -> str:
    try:
        response = httpx.get(SHEET_URL, timeout=5, follow_redirects=True)
        lines = response.text.strip().split("\n")
        # 第一行是 header，第二行是 prompt 內容
        if len(lines) >= 2:
            return lines[1].strip().strip('"')
        return FALLBACK_PROMPT
    except Exception as e:
        logger.error(f"Failed to load prompt from Sheets: {e}")
        return FALLBACK_PROMPT

# ── Router prompt (hardcoded，不需要常改) ────────────────────────────────────
ROUTER_PROMPT = """你是一個訊息分類助手。

判斷使用者的訊息是否包含「需要事實查核的聲明」（例如健康、食安、政治、時事、科學相關的具體聲明，可能為假訊息或轉傳文）。

只回覆以下其中一個詞，不要有其他文字：
- FACT_CHECK （需要查核）
- CHAT （一般閒聊或問題）
"""

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()

bot_user_id: str = ""


@app.on_event("startup")
async def fetch_bot_user_id():
    global bot_user_id
    try:
        with ApiClient(line_config) as api_client:
            api = MessagingApi(api_client)
            profile = api.get_bot_info()
            bot_user_id = profile.user_id
            logger.info(f"Bot user ID: {bot_user_id}")
    except Exception as e:
        logger.error(f"Failed to fetch bot user ID: {e}")


# ── Webhook endpoint ──────────────────────────────────────────────────────────
@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return {"status": "ok"}


# ── Message event handler ─────────────────────────────────────────────────────
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    text: str = event.message.text or ""

    if not is_bot_mentioned(event, text):
        return

    quoted_text, user_instruction = extract_target_text(event, text)
    combined = "\n".join(filter(None, [quoted_text, user_instruction]))

    if not combined:
        reply(event, "請 reply 要查核的訊息後再 @ 我，或是直接把要查核的內容貼在 @ 後面 🙏")
        return

    try:
        # 每次從 Google Sheets 載入最新 prompt
        system_prompt = load_prompt()
        logger.info("Prompt loaded from Sheets.")

        # Step 1: Gemma 判斷要不要查核
        decision = route(combined)
        logger.info(f"Router decision: {decision}")

        if decision == "FACT_CHECK":
            # Step 2a: Gemini + Google Search 查核
            result = fact_check(quoted_text, user_instruction, system_prompt)
        else:
            # Step 2b: Gemma 閒聊回覆
            result = chat(combined, system_prompt)

        reply(event, result)

    except Exception as e:
        logger.error(f"Failed: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            reply(event, "⚠️ API 額度用完了，明天再試 🙏")
        else:
            reply(event, "⚠️ 失敗，快叫開發者來修。")


# ── Helper: detect bot mention ────────────────────────────────────────────────
def is_bot_mentioned(event: MessageEvent, text: str) -> bool:
    mention = getattr(event.message, "mention", None)
    if mention and mention.mentionees:
        for m in mention.mentionees:
            if getattr(m, "user_id", None) == bot_user_id:
                return True
    if "@" in text:
        return True
    return False


# ── Helper: extract quoted + instruction ─────────────────────────────────────
def extract_target_text(event: MessageEvent, text: str) -> tuple[str, str]:
    quoted = getattr(event.message, "quoted_message_preview", None)
    quoted_text = ""
    if quoted and getattr(quoted, "text", None):
        quoted_text = quoted.text.strip()
    user_instruction = re.sub(r"@\S+", "", text).strip()
    return quoted_text, user_instruction


# ── Helper: reply ─────────────────────────────────────────────────────────────
def reply(event: MessageEvent, message: str):
    with ApiClient(line_config) as api_client:
        api = MessagingApi(api_client)
        api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=message)],
            )
        )


# ── Step 1: Gemma 3 27B router ────────────────────────────────────────────────
def route(text: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"{ROUTER_PROMPT}\n\n訊息內容：「{text}」",
    )
    decision = response.text.strip().upper()
    return "FACT_CHECK" if "FACT_CHECK" in decision else "CHAT"


# ── Step 2a: Gemma 3 27B 閒聊 ────────────────────────────────────────────────
def chat(text: str, system_prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemma-3-27b-it",
        contents=f"{system_prompt}\n\n訊息：「{text}」",
    )
    return response.text.strip()


# ── Step 2b: Gemini + Google Search 查核 ─────────────────────────────────────
def fact_check(quoted_text: str, user_instruction: str, system_prompt: str) -> str:
    parts = []
    if quoted_text:
        parts.append(f"查核內容：「{quoted_text}」")
    if user_instruction:
        parts.append(f"使用者補充指示：「{user_instruction}」")

    prompt = system_prompt + "\n\n" + "\n".join(parts)

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )
    return response.text.strip()
