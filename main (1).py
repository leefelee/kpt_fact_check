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
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Load system prompt from file ──────────────────────────────────────────────
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
logger.info("System prompt loaded.")

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

    quoted_text, user_instruction = extract_target_text(event, text)

    if not quoted_text and not user_instruction:
        reply(event, "請 reply 要查核的訊息後再 @ 我，或是直接把要查核的內容貼在 @ 後面 🙏")
        return

    try:
        result = fact_check(quoted_text, user_instruction)
        reply(event, result)
    except Exception as e:
        logger.error(f"Fact check failed: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            reply(event, "⚠️ API 額度用完了，今天查不了，明天再試試看 🙏")
        else:
            reply(event, "⚠️ 失敗，快叫開發者來修。")


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
    quoted = getattr(event.message, "quoted_message_preview", None)
    quoted_text = ""
    if quoted and getattr(quoted, "text", None):
        quoted_text = quoted.text.strip()

    user_instruction = re.sub(r"@\S+", "", text).strip()

    return quoted_text, user_instruction


# ── Helper: reply message ─────────────────────────────────────────────────────
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
def fact_check(quoted_text: str, user_instruction: str) -> str:
    parts = []
    if quoted_text:
        parts.append(f"查核內容：「{quoted_text}」")
    if user_instruction:
        parts.append(f"使用者補充指示：「{user_instruction}」")

    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(parts)

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        ),
    )

    return response.text.strip()
