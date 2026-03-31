import os
import re
import csv
import io
import logging
import httpx
from datetime import datetime, timezone, timedelta
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

# ── Google Sheets Prompt Loader ───────────────────────────────────────────────
SHEET_ID = "1QjU_OVHcEEcxSpVguExAh2rmmVJBK9KLDhksIVRs4vM"
# 注意：Google Sheets 請設定為 Key, Value 兩欄格式，可略過第一行 Header
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# 內建備用 Prompt (當 Google Sheets 讀取失敗或欄位遺失時的保險機制)
FALLBACK_PROMPTS = {
    "SYSTEM_PERSONA": "你是家庭 LINE 群組的晚輩助手「老查」。負責提供精準的即時資訊查詢與事實查核，偶爾陪長輩閒聊。語氣親切簡潔，可加 1 個 emoji。絕對不要稱呼「您」，不要說教。",
    "TASK_ROUTER": "你是一個訊息分類系統。請判斷使用者的訊息是否屬於以下兩種情況之一：\n1. 需事實查核的聲明\n2. 需即時資訊的查詢（如天氣/股價/時間）",
    "TASK_CHAT": "請根據老查的人格設定，回應使用者的閒聊。若使用者只是打招呼，請輕鬆回應。",
    "TASK_FACT_CHECK": "請參考提供的 Google 搜尋結果，判斷使用者訊息的真實性，或是回答使用者的即時資訊查詢。",
    "FORMAT_ROUTER": "[強制指令] 只要符合需聯網查資料的情況，請回覆 SEARCH_NEEDED。若是純閒聊或一般問題，請回覆 CHAT。只回覆這兩個詞之一，無須解釋。",
    "FORMAT_RESPONSE": "[強制指令]\n1. 長度極限：最多 3 行，150字內。\n2. 查核格式：[❌假訊息 / ✅正確 / ⚠️待查證]\\n真相：...\\n來源：...\n3. 查詢格式：直接給答案，不解釋搜尋過程。"
}

def load_prompts() -> dict:
    prompts = FALLBACK_PROMPTS.copy()
    try:
        response = httpx.get(SHEET_URL, timeout=5, follow_redirects=True)
        response.raise_for_status()
        
        # 解析 CSV
        reader = csv.reader(io.StringIO(response.text))
        next(reader, None) # 跳過 Header
        
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                value = row[1].strip()
                if key and value:
                    prompts[key] = value
        logger.info("Prompts loaded successfully from Sheets.")
    except Exception as e:
        logger.error(f"Failed to load prompts from Sheets, using fallbacks. Error: {e}")
    
    return prompts

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
        reply(event, "請 reply 要處理的訊息後再 @ 我，或是直接把內容貼在 @ 後面 🙏")
        return

    try:
        # 取得最新 Prompts 字典
        prompts = load_prompts()

        # Step 1: 判斷任務路由
        decision = route(combined, prompts)
        logger.info(f"Router decision: {decision}")

        # Step 2: 執行對應任務
        if decision == "SEARCH_NEEDED":
            result = fact_check(quoted_text, user_instruction, prompts)
        else:
            result = chat(combined, prompts)

        # 最終防線：長度截斷
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

def get_current_time_context() -> str:
    # 固定回傳 UTC+8 台灣時間，作為 Prompt 的動態脈絡 (L3)
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    return f"【系統當前時間：{now.strftime('%Y年%m月%d日 %H:%M')}】"

# ── Generation functions ──────────────────────────────────────────────────────
def route(text: str, prompts: dict) -> str:
    # L2 + L3 + L4
    prompt_parts = [
        prompts.get("TASK_ROUTER", ""),
        f"使用者訊息：「{text}」",
        prompts.get("FORMAT_ROUTER", "")
    ]
    full_prompt = "\n\n".join(filter(None, prompt_parts))
    
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=full_prompt,
    )
    decision = response.text.strip().upper()
    return "SEARCH_NEEDED" if "SEARCH_NEEDED" in decision else "CHAT"

def chat(text: str, prompts: dict) -> str:
    # L1 + L2 + L3 + L4
    prompt_parts = [
        prompts.get("SYSTEM_PERSONA", ""),
        prompts.get("TASK_CHAT", ""),
        get_current_time_context(),
        f"使用者訊息：「{text}」",
        prompts.get("FORMAT_RESPONSE", "")
    ]
    full_prompt = "\n\n".join(filter(None, prompt_parts))

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=150,
            temperature=0.7
        )
    )
    return response.text.strip()

def fact_check(quoted_text: str, user_instruction: str, prompts: dict) -> str:
    # L1 + L2 + L3 + L4
    prompt_parts = [
        prompts.get("SYSTEM_PERSONA", ""),
        prompts.get("TASK_FACT_CHECK", ""),
        get_current_time_context()
    ]
    
    if quoted_text:
        prompt_parts.append(f"待處理內容：「{quoted_text}」")
    if user_instruction:
        prompt_parts.append(f"使用者指示：「{user_instruction}」")
        
    prompt_parts.append(prompts.get("FORMAT_RESPONSE", ""))
    
    full_prompt = "\n\n".join(filter(None, prompt_parts))

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=200, 
            temperature=0.2 
        ),
    )
    return response.text.strip()
