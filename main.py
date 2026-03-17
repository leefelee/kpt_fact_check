import os
import re
import logging
from google import genai
from google.genai import types

from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    PushMessageRequest,
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
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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

    # Extract quoted content + user instruction separately
    quoted_text, user_instruction = extract_target_text(event, text)

    source_id = get_source_id(event)

    if not quoted_text and not user_instruction:
        push(source_id, "請 reply 要查核的訊息後再 @ 我，或是直接把要查核的內容貼在 @ 後面 🙏")
        return

    push(source_id, "🔍 等等喔，我想想")

    try:
        result = fact_check(quoted_text, user_instruction)
        push(source_id, result)
    except Exception as e:
        logger.error(f"Fact check failed: {e}")
        push(source_id, "⚠️ 失敗，請稍後再試。")


# ── Helper: detect bot mention ────────────────────────────────────────────────
def is_bot_mentioned(event: MessageEvent, text: str) -> bool:
    mention = getattr(event.message, "mention", None)
    if mention and mention.mentionees:
        for m in mention.mentionees:
            uid = getattr(m, "user_id", None)
            if uid == bot_user_id:
                return True

    if "@" in text:
        return True

    return False


# ── Helper: extract quoted text + user instruction ────────────────────────────
def extract_target_text(event: MessageEvent, text: str) -> tuple[str, str]:
    # Get quoted (reply) message content
    quoted = getattr(event.message, "quoted_message_preview", None)
    quoted_text = ""
    if quoted and getattr(quoted, "text", None):
        quoted_text = quoted.text.strip()

    # Get text typed after @mention (strip all @Xxx tokens)
    user_instruction = re.sub(r"@\S+", "", text).strip()

    return quoted_text, user_instruction


# ── Helper: get push target ID ────────────────────────────────────────────────
def get_source_id(event: MessageEvent) -> str:
    source = event.source
    if hasattr(source, "group_id") and source.group_id:
        return source.group_id
    if hasattr(source, "room_id") and source.room_id:
        return source.room_id
    return source.user_id


# ── Helper: push message ──────────────────────────────────────────────────────
def push(to: str, message: str):
    with ApiClient(line_config) as api_client:
        api = MessagingApi(api_client)
        api.push_message(
            PushMessageRequest(
                to=to,
                messages=[TextMessage(text=message)],
            )
        )


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一個專業的即時事實查核助手，專門協助台灣用戶尤其是不熟悉科技的長輩，辨別 LINE 群組中流傳的假訊息。

你會先辨識使用者給你的資訊、問題是否有「需要請你協助查核的命題」，如果有請依照以下格式回覆：

----------------------------------------------
【查核結果】❌ 假訊息 / ✅ 正確 / ⚠️ 待查證

【問題所在】
（如為假訊息：說明哪裡錯誤、為何是假的）

【正確資訊】
（提供正確的說法或背景知識）

【來源】
1. 來源名稱：網址
2. 來源名稱：網址
-----------------------------------------------
注意事項：
- 用繁體中文回覆
- 語氣親切，適合家庭群組，不需要用您
- 來源盡量引用台灣媒體、政府官方網站、或國際可信媒體
- 若資訊不足以判斷，誠實標示「待查證」並說明原因
- 回覆長度控制在 300 字元以內
- **如果使用者提供給你的文字判斷沒有「需要請你協助查核的命題」，則以一般 gpt 的形式回覆內容，但將回覆長度控制在 100 字元之內

"""


# ── Core: Gemini fact-check ───────────────────────────────────────────────────
def fact_check(quoted_text: str, user_instruction: str) -> str:
    parts = []
    if quoted_text:
        parts.append(f"查核內容：「{quoted_text}」")
    if user_instruction:
        parts.append(f"使用者補充指示：「{user_instruction}」")

    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(parts)

    response = gemini_client.models.generate_content(
        model="gemini-3-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )

    return response.text.strip()
