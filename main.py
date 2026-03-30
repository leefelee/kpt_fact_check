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

# ── System Prompts & Constraints ──────────────────────────────────────────────
ROUTER_PROMPT = """你是一個訊息分類助手。

判斷使用者的訊息是否屬於以下兩種情況之一：
1. 「需要事實查核的聲明」（例如健康、食安、政治、時事的具體聲明或轉傳文）
2. 「需要即時資訊的查詢」（例如詢問今天天氣、股價、新聞、當前時間）

只要符合上述任一情況（需要聯網查資料），請回覆 SEARCH_NEEDED。
若是純粹閒聊、打招呼或無需聯網的一般問題，請回覆 CHAT。

只回覆以下其中一個詞，不要有其他文字：
- SEARCH_NEEDED
- CHAT
"""

# 全域強制指令，確保模型在所有情境下壓縮輸出
HARD_CONSTRAINT = "\n\n[系統強制指令]：回覆請極度簡練，絕對不可超過 3 行，總字數必須控制在 150 字以內。"

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
        reply(event, "請 reply 要查核的訊息後再 @ 我，或是直接把要查詢的內容貼在 @ 後面 🙏")
        return

    try:
        system_prompt = load_prompt()
        logger.info("Prompt loaded from Sheets.")

        decision = route(combined)
        logger.info(f"Router decision: {decision}")

        if decision == "SEARCH_NEEDED":
            result = fact_check(quoted_text, user_instruction, system_prompt)
        else:
            result = chat(combined, system_prompt)

        # 最終防線：Python 層級字串截斷
        final_result = truncate_text(result, max_length=150)
        
        reply(event, final_result)

    except Exception as e:
        logger.error(f"Failed: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            reply(event, "⚠️ API 額度用完了，明天再試 🙏")
        else:
            reply(event, "⚠️ 系統發生錯誤。")

# ── Helper functions ──────────────────────────────────────────────────────────
def is_bot_mentioned(event: MessageEvent, text: str) -> bool:
    mention = getattr(event.message, "mention", None)
    if mention and mention.mentionees:
        for m in mention.mentionees:
            if getattr(m, "user_id", None) == bot_user_id:
                return True
    return False

def extract_target_text(event: MessageEvent, text: str) -> tuple[str, str]:
    quoted = getattr(event.message, "quoted_message_preview", None)
    quoted_text = ""
    if quoted and getattr(quoted, "text", None):
        quoted_text = quoted.text.strip()
    user_instruction = re.sub(r"@\S+", "", text).strip()
    return quoted_text, user_instruction

def truncate_text(text: str, max_length: int = 150) -> str:
    if len(text) > max_length:
        return text[:max_length] + "...\n(字數達上限)"
    return text

def reply(event: MessageEvent, message: str):
    with ApiClient(line_config) as api_client:
        api = MessagingApi(api_client)
        api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=message)],
            )
        )

# ── Generation functions ──────────────────────────────────────────────────────
def route(text: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"{ROUTER_PROMPT}\n\n訊息內容：「{text}」",
    )
    decision = response.text.strip().upper()
    return "SEARCH_NEEDED" if "SEARCH_NEEDED" in decision else "CHAT"

def chat(text: str, system_prompt: str) -> str:
    prompt = f"{system_prompt}\n\n訊息：「{text}」{HARD_CONSTRAINT}"
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=150,
            temperature=0.7
        )
    )
    return response.text.strip()

def fact_check(quoted_text: str, user_instruction: str, system_prompt: str) -> str:
    parts = []
    if quoted_text:
        parts.append(f"處理內容：「{quoted_text}」")
    if user_instruction:
        parts.append(f"使用者補充指示：「{user_instruction}」")

    prompt = system_prompt + "\n\n" + "\n".join(parts) + HARD_CONSTRAINT

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=200, 
            temperature=0.2 
        ),
    )
    return response.text.strip()
