import os
import re
import logging
import google.generativeai as genai

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

# ── Gemini setup ──────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

GOOGLE_SEARCH_TOOL = genai.protos.Tool(
    google_search_retrieval=genai.protos.GoogleSearchRetrieval(
        dynamic_retrieval_config=genai.protos.DynamicRetrievalConfig(
            mode=genai.protos.DynamicRetrievalConfig.Mode.MODE_DYNAMIC,
            dynamic_threshold=0.3,
        )
    )
)

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


# ── Bot user ID (fetched once on startup) ─────────────────────────────────────
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

    # Only respond when bot is mentioned
    if not is_bot_mentioned(event, text):
        return

    # Extract the content to fact-check
    target_text = extract_target_text(event, text)

    if not target_text:
        reply(event, "請 reply 要查核的訊息後再 @ 我，或是直接把要查核的內容貼在 @ 後面 🙏")
        return

    reply(event, "🔍 查核中，請稍候...")

    try:
        result = fact_check(target_text)
        reply(event, result)
    except Exception as e:
        logger.error(f"Fact check failed: {e}")
        reply(event, "⚠️ 查核失敗，請稍後再試。")


# ── Helper: detect bot mention ────────────────────────────────────────────────
def is_bot_mentioned(event: MessageEvent, text: str) -> bool:
    # Check mention object (group chats provide this)
    mention = getattr(event.message, "mention", None)
    if mention and mention.mentionees:
        for m in mention.mentionees:
            uid = getattr(m, "user_id", None)
            if uid == bot_user_id:
                return True

    # Fallback: check if text contains any @mention pattern
    # (covers edge cases where mention object isn't populated)
    if "@" in text:
        return True

    return False


# ── Helper: extract the text to check ────────────────────────────────────────
def extract_target_text(event: MessageEvent, text: str) -> str:
    # Priority 1: quoted (reply) message content
    quoted = getattr(event.message, "quoted_message_preview", None)
    if quoted and getattr(quoted, "text", None):
        return quoted.text.strip()

    # Priority 2: text after @mention in the current message
    # Strip all @Xxx mentions from message and use the remainder
    clean = re.sub(r"@\S+", "", text).strip()
    if clean:
        return clean

    return ""


# ── Helper: send reply ────────────────────────────────────────────────────────
def reply(event: MessageEvent, message: str):
    with ApiClient(line_config) as api_client:
        api = MessagingApi(api_client)
        api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=message)],
            )
        )


# ── Core: Gemini fact-check ───────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一個專業的即時事實查核助手，專門協助台灣用戶辨別 LINE 群組中流傳的假訊息。

請依照以下格式回覆：

【查核結果】❌ 假訊息 / ✅ 正確 / ⚠️ 待查證

【問題所在】
（如為假訊息：說明哪裡錯誤、為何是假的）

【正確資訊】
（提供正確的說法或背景知識）

【來源】
1. 來源名稱：網址
2. 來源名稱：網址

注意事項：
- 用繁體中文回覆
- 語氣親切，適合家庭群組
- 來源盡量引用台灣媒體、政府官方網站、或國際可信媒體
- 若資訊不足以判斷，誠實標示「待查證」並說明原因
- 回覆長度控制在 300 字以內
"""


def fact_check(text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n請查核以下內容：\n\n「{text}」"

    response = gemini_model.generate_content(
        prompt,
        tools=[GOOGLE_SEARCH_TOOL],
    )

    return response.text.strip()
