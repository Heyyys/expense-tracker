import streamlit as st

# set_page_config MUST be the first Streamlit command
st.set_page_config(page_title="Expense Tracker AI Agent", layout="centered")

import pandas as pd
import sqlite3
from datetime import datetime
import os
import re
import json
import hashlib
import requests
import easyocr
import fitz  # PyMuPDF — for PDF to image conversion
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import whisper
import torch

# ========================
# Database (shared for all users)
# ========================
_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expenses.db')
conn = sqlite3.connect(_db_path, check_same_thread=False)

# Users table
conn.execute('''CREATE TABLE IF NOT EXISTS users
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password_hash TEXT NOT NULL,
                 created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

# Expenses table (with username)
conn.execute('''CREATE TABLE IF NOT EXISTS expenses
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT NOT NULL,
                 date TEXT,
                 merchant TEXT,
                 category TEXT,
                 currency TEXT DEFAULT 'HKD',
                 amount REAL,
                 amount_hkd REAL,
                 items TEXT,
                 source TEXT)''')
conn.commit()

# Migrate: add username column if missing (for existing DBs)
try:
    conn.execute("ALTER TABLE expenses ADD COLUMN username TEXT DEFAULT ''")
    conn.commit()
except sqlite3.OperationalError:
    pass
try:
    conn.execute("ALTER TABLE expenses ADD COLUMN currency TEXT DEFAULT 'HKD'")
    conn.commit()
except sqlite3.OperationalError:
    pass
try:
    conn.execute("ALTER TABLE expenses ADD COLUMN amount_hkd REAL")
    conn.commit()
except sqlite3.OperationalError:
    pass

# ========================
# Auth helpers
# ========================
def _hash_password(password: str) -> str:
    """Hash password with SHA-256 + salt."""
    salt = "expense_tracker_2026"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def register_user(username: str, password: str) -> bool:
    try:
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                     (username.strip().lower(), _hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

def authenticate_user(username: str, password: str) -> bool:
    row = conn.execute("SELECT password_hash FROM users WHERE username = ?",
                       (username.strip().lower(),)).fetchone()
    if row and row[0] == _hash_password(password):
        return True
    return False

# ========================
# i18n — Translation
# ========================
TRANSLATIONS = {
    "en": {
        "page_title": "Expense Tracker AI Agent",
        "main_title": "Personal Expense Tracker AI Agent",
        "main_desc": "Upload a receipt photo, record voice, or type text — the AI agent will extract and save your expense!",
        "tab_photo": "Photo Receipt",
        "tab_voice": "Voice Input",
        "tab_free": "Free Text",
        "upload_label": "Upload a receipt (JPG/PNG/PDF)",
        "extracted_text": "**Extracted Text:**",
        "btn_parse_photo": "Parse with AI Agent & Save (Photo)",
        "btn_parse_voice": "Parse with AI Agent & Save (Voice)",
        "voice_label": "Record your expense (e.g., 'Coffee at Starbucks 150 dollars today')",
        "spinner_ocr": "Performing OCR...",
        "spinner_transcribe": "Transcribing voice with Whisper...",
        "spinner_ai": "AI Agent analyzing...",
        "transcribed_text": "**Transcribed Text:**",
        "free_text_label": "Type freely (e.g., 'Starbucks coffee 150 HKD')",
        "free_text_btn": "Parse",
        "free_text_confirm": "Confirm & Save",
        "free_text_saved": "Saved — zero API calls!",
        "success_added": "Added: **{merchant}** — {amount} {currency} = {amount_hkd:.2f} HKD ({category}) on {date}",
        "parsed_local": "Parsed locally (free)",
        "parsed_api": "Parsed via API",
        "header_all_expenses": "All Recorded Expenses",
        "header_monthly": "Monthly Summary",
        "select_month": "Select month",
        "metric_total": "Total (HKD)",
        "metric_transactions": "Transactions",
        "metric_avg_txn": "Avg / Transaction",
        "metric_avg_day": "Avg / Day",
        "sub_category": "Spending by Category — {month}",
        "sub_daily": "Daily Spending — {month}",
        "sub_merchants": "Top Merchants — {month}",
        "sub_currency": "Currency Breakdown — {month}",
        "col_category": "Category",
        "col_amount_hkd": "Amount (HKD)",
        "col_merchant": "Merchant",
        "col_total_hkd": "Total (HKD)",
        "col_visits": "Visits",
        "col_currency": "Currency",
        "no_expenses_month": "No expenses for {month}.",
        "no_expenses_yet": "No expenses recorded yet. Add one above!",
        "sidebar_lang": "Language",
        "sidebar_fx_header": "FX Rates (to HKD)",
        "sidebar_fx_live": "Rates auto-updated daily from open.er-api.com. You can override below.",
        "sidebar_fx_fallback": "Using fallback rates (offline). Edit manually below.",
        "sidebar_fx_refresh": "Refresh rates now",
        "tab_quick": "Quick Form",
        "quick_date": "Date",
        "quick_merchant": "Merchant / Store",
        "quick_currency": "Currency",
        "quick_amount": "Amount",
        "quick_category": "Category",
        "quick_items": "Items / Description",
        "quick_submit": "Save Expense",
        "quick_success": "Saved: **{merchant}** — {amount} {currency} = {amount_hkd:.2f} HKD ({category}) on {date}",
        "quick_no_api": "No API call (direct entry)",
        "text_preview": "Preview",
        "footer": "Built with Streamlit + EasyOCR + Whisper + Grok (via LangChain) — Perfect for daily expense tracking!",
        # Auth strings
        "login_title": "Login",
        "register_title": "Register",
        "username": "Username",
        "password": "Password",
        "password_confirm": "Confirm Password",
        "btn_login": "Log In",
        "btn_register": "Create Account",
        "login_success": "Welcome back, **{user}**!",
        "register_success": "Account created! You are now logged in as **{user}**.",
        "login_error": "Invalid username or password.",
        "register_error_exists": "Username already taken. Please choose another.",
        "register_error_mismatch": "Passwords do not match.",
        "register_error_short": "Password must be at least 4 characters.",
        "register_error_empty": "Please fill in all fields.",
        "logout": "Logout",
        "logged_in_as": "Logged in as",
        "switch_to_register": "Don't have an account? Register",
        "switch_to_login": "Already have an account? Login",
        "delete_selected": "Delete Selected",
        "delete_success": "{count} expense(s) deleted.",
        "delete_none": "No rows selected. Tick the checkboxes on the left to select expenses to delete.",
        "col_select": "Select",
        "btn_parse_review": "Parse Receipt",
        "btn_parse_voice_review": "Parse Voice",
        "review_header": "Review & Edit before saving",
        "review_save": "Confirm & Save",
        "review_saved": "Saved!",
        "review_parsed_local": "Parsed locally (free) — review below",
        "review_parsed_api": "Parsed via API — review below",
        "save_changes": "Save Changes",
        "save_changes_success": "{count} expense(s) updated.",
        "save_changes_none": "No changes detected.",
    },
    "zh-TW": {
        "page_title": "AI 記帳助手",
        "main_title": "個人 AI 記帳助手",
        "main_desc": "上傳收據照片、語音錄入或手動輸入文字 — AI 助手會自動擷取並儲存你的支出！",
        "tab_photo": "拍照收據",
        "tab_voice": "語音輸入",
        "tab_free": "自由輸入",
        "upload_label": "上傳收據（JPG/PNG/PDF）",
        "extracted_text": "**擷取文字：**",
        "btn_parse_photo": "AI 解析並儲存（照片）",
        "btn_parse_voice": "AI 解析並儲存（語音）",
        "voice_label": "錄製你的消費（例如：「星巴克咖啡 150 元」）",
        "spinner_ocr": "正在 OCR 辨識中...",
        "spinner_transcribe": "正在用 Whisper 轉錄語音...",
        "spinner_ai": "AI 助手分析中...",
        "transcribed_text": "**轉錄文字：**",
        "free_text_label": "自由輸入（例如：「星巴克 咖啡 150 HKD」）",
        "free_text_btn": "解析",
        "free_text_confirm": "確認並儲存",
        "free_text_saved": "已儲存 — 未使用 API！",
        "success_added": "已新增：**{merchant}** — {amount} {currency} = {amount_hkd:.2f} HKD（{category}）{date}",
        "parsed_local": "本地解析（免費）",
        "parsed_api": "透過 API 解析",
        "header_all_expenses": "所有支出記錄",
        "header_monthly": "月度摘要",
        "select_month": "選擇月份",
        "metric_total": "總計（HKD）",
        "metric_transactions": "交易筆數",
        "metric_avg_txn": "每筆平均",
        "metric_avg_day": "每日平均",
        "sub_category": "分類支出 — {month}",
        "sub_daily": "每日支出 — {month}",
        "sub_merchants": "常去商家 — {month}",
        "sub_currency": "幣別分佈 — {month}",
        "col_category": "分類",
        "col_amount_hkd": "金額（HKD）",
        "col_merchant": "商家",
        "col_total_hkd": "總計（HKD）",
        "col_visits": "次數",
        "col_currency": "幣別",
        "no_expenses_month": "{month} 尚無支出記錄。",
        "no_expenses_yet": "尚無支出記錄，請在上方新增！",
        "sidebar_lang": "語言",
        "sidebar_fx_header": "匯率（換算 HKD）",
        "sidebar_fx_live": "匯率每日自動更新自 open.er-api.com，可手動覆寫。",
        "sidebar_fx_fallback": "目前使用離線匯率，請手動編輯。",
        "sidebar_fx_refresh": "立即更新匯率",
        "tab_quick": "快速表單",
        "quick_date": "日期",
        "quick_merchant": "商家 / 店名",
        "quick_currency": "幣別",
        "quick_amount": "金額",
        "quick_category": "分類",
        "quick_items": "品項 / 說明",
        "quick_submit": "儲存支出",
        "quick_success": "已儲存：**{merchant}** — {amount} {currency} = {amount_hkd:.2f} HKD（{category}）{date}",
        "quick_no_api": "未呼叫 API（直接輸入）",
        "text_preview": "預覽",
        "footer": "使用 Streamlit + EasyOCR + Whisper + Grok（LangChain）打造 — 日常記帳好幫手！",
        # Auth strings
        "login_title": "登入",
        "register_title": "註冊",
        "username": "帳號",
        "password": "密碼",
        "password_confirm": "確認密碼",
        "btn_login": "登入",
        "btn_register": "建立帳號",
        "login_success": "歡迎回來，**{user}**！",
        "register_success": "帳號建立成功！你已登入為 **{user}**。",
        "login_error": "帳號或密碼錯誤。",
        "register_error_exists": "帳號已被使用，請換一個。",
        "register_error_mismatch": "兩次輸入的密碼不一致。",
        "register_error_short": "密碼至少需要 4 個字元。",
        "register_error_empty": "請填寫所有欄位。",
        "logout": "登出",
        "logged_in_as": "目前登入",
        "switch_to_register": "還沒有帳號？註冊",
        "switch_to_login": "已有帳號？登入",
        "delete_selected": "刪除所選",
        "delete_success": "已刪除 {count} 筆支出。",
        "delete_none": "未選擇任何項目。請勾選左側核取方塊以選擇要刪除的支出。",
        "col_select": "選取",
        "btn_parse_review": "解析收據",
        "btn_parse_voice_review": "解析語音",
        "review_header": "確認並編輯後儲存",
        "review_save": "確認並儲存",
        "review_saved": "已儲存！",
        "review_parsed_local": "本地解析（免費）— 請在下方檢查",
        "review_parsed_api": "透過 API 解析 — 請在下方檢查",
        "save_changes": "儲存修改",
        "save_changes_success": "已更新 {count} 筆支出。",
        "save_changes_none": "未偵測到任何變更。",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def t(key: str, **kwargs) -> str:
    text = TRANSLATIONS.get(st.session_state.lang, TRANSLATIONS["en"]).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

# ========================
# Sidebar — Language (always visible)
# ========================
with st.sidebar:
    st.subheader(f"🌐 {t('sidebar_lang')}")
    lang_choice = st.radio(
        t("sidebar_lang"),
        options=["English", "繁體中文"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    new_lang = "en" if lang_choice == "English" else "zh-TW"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

# ========================
# Auth Gate — Login / Register
# ========================
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if st.session_state.logged_in_user is None:
    st.title(f"🧾 {t('main_title')}")

    if st.session_state.auth_mode == "login":
        st.subheader(f"🔐 {t('login_title')}")
        with st.form("login_form"):
            login_user = st.text_input(t("username"))
            login_pass = st.text_input(t("password"), type="password")
            if st.form_submit_button(t("btn_login")):
                if login_user and login_pass:
                    if authenticate_user(login_user, login_pass):
                        st.session_state.logged_in_user = login_user.strip().lower()
                        st.rerun()
                    else:
                        st.error(t("login_error"))
                else:
                    st.error(t("register_error_empty"))
        if st.button(t("switch_to_register")):
            st.session_state.auth_mode = "register"
            st.rerun()

    else:  # register mode
        st.subheader(f"📝 {t('register_title')}")
        with st.form("register_form"):
            reg_user = st.text_input(t("username"))
            reg_pass = st.text_input(t("password"), type="password")
            reg_pass2 = st.text_input(t("password_confirm"), type="password")
            if st.form_submit_button(t("btn_register")):
                if not reg_user or not reg_pass:
                    st.error(t("register_error_empty"))
                elif len(reg_pass) < 4:
                    st.error(t("register_error_short"))
                elif reg_pass != reg_pass2:
                    st.error(t("register_error_mismatch"))
                else:
                    if register_user(reg_user, reg_pass):
                        st.session_state.logged_in_user = reg_user.strip().lower()
                        st.success(t("register_success", user=reg_user.strip().lower()))
                        st.rerun()
                    else:
                        st.error(t("register_error_exists"))
        if st.button(t("switch_to_login")):
            st.session_state.auth_mode = "login"
            st.rerun()

    st.stop()  # Don't render the rest of the app until logged in

# ========================
# User is logged in — show main app
# ========================
CURRENT_USER = st.session_state.logged_in_user

# Sidebar: user info + logout
with st.sidebar:
    st.divider()
    st.write(f"👤 **{t('logged_in_as')}:** {CURRENT_USER}")
    if st.button(f"🚪 {t('logout')}"):
        st.session_state.logged_in_user = None
        st.rerun()

# ========================
# Load environment variables
# ========================
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_env_path, override=True)

if not os.getenv("XAI_API_KEY") and os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("XAI_API_KEY=") or _line.startswith("xAI_API_KEY="):
                os.environ["XAI_API_KEY"] = _line.split("=", 1)[1]

# ========================
# Device detection
# ========================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ========================
# Model loading (cached)
# ========================
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en', 'ch_tra'], gpu=torch.cuda.is_available())

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base", device=DEVICE)

reader = load_ocr_reader()
whisper_model = load_whisper_model()

# ========================
# FX Rates
# ========================
FALLBACK_FX_RATES = {
    "HKD": 1.0, "TWD": 4.12, "USD": 0.128, "CNY": 0.93, "JPY": 19.5,
    "EUR": 0.118, "GBP": 0.1, "SGD": 0.17, "KRW": 178.0, "MYR": 0.57,
}
SUPPORTED_CURRENCIES = ["HKD", "TWD", "USD", "CNY", "JPY", "EUR", "GBP", "SGD", "KRW", "MYR"]

@st.cache_data(ttl=86400)
def fetch_fx_rates() -> dict:
    try:
        resp = requests.get("https://open.er-api.com/v6/latest/HKD", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "success":
            live = {cur: data["rates"].get(cur, FALLBACK_FX_RATES.get(cur, 1.0))
                    for cur in SUPPORTED_CURRENCIES}
            live["HKD"] = 1.0
            return live
    except Exception:
        pass
    return FALLBACK_FX_RATES.copy()

_live_rates = fetch_fx_rates()

if "fx_rates" not in st.session_state:
    st.session_state.fx_rates = _live_rates.copy()
if "fx_source" not in st.session_state:
    st.session_state.fx_source = "live" if _live_rates != FALLBACK_FX_RATES else "fallback"

def convert_to_hkd(amount: float, currency: str) -> float:
    rate = st.session_state.fx_rates.get(currency.upper(), None)
    if rate is None or rate == 0:
        return amount
    return round(amount / rate, 2)

# Sidebar: FX rates
with st.sidebar:
    st.header(f"💱 {t('sidebar_fx_header')}")
    if st.session_state.fx_source == "live":
        st.caption(t("sidebar_fx_live"))
    else:
        st.caption(t("sidebar_fx_fallback"))

    if st.button(f"🔄 {t('sidebar_fx_refresh')}"):
        st.cache_data.clear()
        fresh = fetch_fx_rates()
        st.session_state.fx_rates = fresh.copy()
        st.session_state.fx_source = "live" if fresh != FALLBACK_FX_RATES else "fallback"
        st.rerun()

    for cur in SUPPORTED_CURRENCIES:
        if cur == "HKD":
            continue
        new_rate = st.number_input(
            f"1 HKD = ? {cur}",
            value=st.session_state.fx_rates[cur],
            min_value=0.0001,
            step=0.01,
            format="%.4f",
            key=f"fx_{cur}",
        )
        st.session_state.fx_rates[cur] = new_rate

# ========================
# Expense model
# ========================
class Expense(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    merchant: str = Field(description="Store or merchant name")
    category: str = Field(description="Category like Food, Transport, Shopping, Entertainment, etc.")
    currency: str = Field(default="HKD", description="Currency code")
    amount: float = Field(description="Total amount in the original currency")
    items: str = Field(description="Brief description of items purchased")

# ========================
# Local parsing helpers
# ========================
CATEGORY_KEYWORDS = {
    "Food": ["coffee", "cafe", "restaurant", "lunch", "dinner", "breakfast", "food", "eat",
             "starbucks", "mcdonald", "kfc", "subway", "pizza", "sushi", "ramen", "bento",
             "飯", "餐", "吃", "咖啡", "早餐", "午餐", "晚餐", "小吃", "便當", "麵"],
    "Transport": ["uber", "taxi", "bus", "mrt", "train", "gas", "parking", "grab",
                  "計程車", "捷運", "公車", "加油", "停車", "高鐵", "火車", "交通"],
    "Shopping": ["shop", "mall", "clothes", "amazon", "uniqlo", "nike", "adidas",
                 "買", "購物", "衣服", "商場", "百貨"],
    "Entertainment": ["movie", "game", "netflix", "spotify", "concert", "bar",
                      "電影", "遊戲", "演唱會", "KTV", "娛樂"],
    "Groceries": ["supermarket", "grocery", "market", "costco", "carrefour", "pxmart",
                  "超市", "全聯", "家樂福", "好市多", "市場", "菜"],
    "Utilities": ["electric", "water", "internet", "phone", "bill",
                  "電費", "水費", "網路", "電話", "帳單"],
    "Health": ["hospital", "doctor", "pharmacy", "medicine", "clinic",
               "醫院", "診所", "藥", "看診", "掛號"],
}

CURRENCY_PATTERNS = {
    "TWD": [r'\bTWD\b', r'\bNT\$', r'\bNT\b', r'元', r'塊', r'台幣'],
    "HKD": [r'\bHKD\b', r'\bHK\$', r'港幣', r'港元'],
    "USD": [r'\bUSD\b', r'\bUS\$', r'\bUS\s*dollars?\b'],
    "CNY": [r'\bCNY\b', r'\bRMB\b', r'人民幣'],
    "JPY": [r'\bJPY\b', r'円', r'日元', r'日幣'],
    "EUR": [r'\bEUR\b', r'€'],
    "GBP": [r'\bGBP\b', r'£'],
    "SGD": [r'\bSGD\b', r'\bSG\$'],
    "KRW": [r'\bKRW\b', r'원', r'韓元'],
    "MYR": [r'\bMYR\b', r'\bRM\b'],
}

def guess_category(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "Other"

def detect_currency(text: str) -> str:
    for currency, patterns in CURRENCY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return currency
    return "HKD"

def _extract_receipt_total(text: str) -> float | None:
    patterns = [
        r'(?:TOTAL|Grand\s*Total|Amount\s*Due|合計|總計|小計|應付|總額)\s*[:\s]*(?:NT\$?|HK\$?|US\$?|\$)?\s*(\d[\d,]*\.?\d*)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    return None

def _extract_receipt_merchant(text: str) -> str | None:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        first = lines[0]
        if not re.match(r'^[\d\s/:.\-]+$', first) and len(first) <= 40:
            return first
    return None

def try_local_parse(text: str) -> Expense | None:
    today = datetime.now().strftime('%Y-%m-%d')
    currency = detect_currency(text)
    is_multiline = '\n' in text

    amount = None
    if is_multiline:
        amount = _extract_receipt_total(text)

    if amount is None:
        nl_match = re.search(r'(?:spent|paid|花了|付了|消費)\s*(?:NT\$?|HK\$?|US\$?|\$)?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if nl_match:
            amount = float(nl_match.group(1))

    if amount is None:
        amount_match = re.search(
            r'(?:NT\$?|HK\$?|US\$?|SG\$?|RM|€|£|\$)\s*(\d+(?:\.\d+)?)'
            r'|(\d+(?:\.\d+)?)\s*(?:TWD|HKD|USD|CNY|JPY|EUR|GBP|SGD|KRW|MYR|元|dollars?|塊|円|원)',
            text, re.IGNORECASE
        )
        if amount_match:
            amount = float(amount_match.group(1) or amount_match.group(2))

    if amount is None:
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        amounts = [float(n) for n in numbers if 1 <= float(n) <= 100000 and len(n) <= 6]
        if len(amounts) == 1:
            amount = amounts[0]
        elif len(amounts) > 1:
            amount = max(amounts)

    if amount is None:
        return None

    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    date = date_match.group(1) if date_match else today

    if is_multiline:
        merchant = _extract_receipt_merchant(text) or "Unknown"
        items = merchant
        category = guess_category(text)
        return Expense(date=date, merchant=merchant, category=category, currency=currency, amount=amount, items=items)

    at_match = re.search(r'(?:at|from|在)\s+(.+?)(?:\s+(?:for|spent|paid|花|付|\d))', text, re.IGNORECASE)
    if at_match:
        merchant = at_match.group(1).strip()
        category = guess_category(text)
        items_text = re.sub(re.escape(merchant), '', text, flags=re.IGNORECASE).strip()
        items_text = re.sub(r'(?:NT\$?|HK\$?|US\$?|SG\$?|RM|€|£|\$)?\s*\d+(?:\.\d+)?\s*(?:TWD|HKD|USD|CNY|JPY|EUR|GBP|SGD|KRW|MYR|元|dollars?|塊|円|원)?', '', items_text, flags=re.IGNORECASE)
        items_text = re.sub(r'\b(?:spent|paid|bought|at|from|for|on|today|yesterday|I|在|花了|付了|消費|買了)\b', '', items_text, flags=re.IGNORECASE).strip().strip('—-,. ')
        items = items_text if items_text else merchant
        return Expense(date=date, merchant=merchant, category=category, currency=currency, amount=amount, items=items)

    remaining = text
    remaining = re.sub(
        r'(?:NT\$?|HK\$?|US\$?|SG\$?|RM|€|£|\$)\s*\d+(?:\.\d+)?'
        r'|\d+(?:\.\d+)?\s*(?:TWD|HKD|USD|CNY|JPY|EUR|GBP|SGD|KRW|MYR|元|dollars?|塊|円|원)',
        '', remaining, flags=re.IGNORECASE
    )
    remaining = re.sub(r'\d{4}-\d{2}-\d{2}', '', remaining)
    remaining = re.sub(r'\b(?:on|at|for|spent|paid|bought|today|yesterday|I|from)\b', '', remaining, flags=re.IGNORECASE)
    remaining = re.sub(r'(?:花了|付了|消費|買了|在)', '', remaining)
    remaining = re.sub(r'\b(?:TWD|HKD|USD|CNY|JPY|EUR|GBP|SGD|KRW|MYR)\b', '', remaining, flags=re.IGNORECASE)
    remaining = remaining.strip().strip('—-,.')

    if not remaining:
        return None

    words = remaining.split()
    if len(words) <= 2:
        merchant = remaining
        items = remaining
    else:
        merchant = words[0]
        items = ' '.join(words[1:])

    category = guess_category(text)
    return Expense(date=date, merchant=merchant, category=category, currency=currency, amount=amount, items=items)

# ========================
# LLM (API fallback)
# ========================
llm = ChatOpenAI(
    model="grok-3-mini-fast",
    temperature=0,
    api_key=os.getenv("XAI_API_KEY") or os.getenv("xAI_API_KEY") or st.secrets.get("XAI_API_KEY", None),
    base_url="https://api.x.ai/v1",
)

if "parse_cache" not in st.session_state:
    st.session_state.parse_cache = {}
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "local_parse_count" not in st.session_state:
    st.session_state.local_parse_count = 0
if "cache_hit_count" not in st.session_state:
    st.session_state.cache_hit_count = 0

def _log_stats(method: str, text: str, expense):
    now = datetime.now().strftime('%H:%M:%S')
    api = st.session_state.api_call_count
    local = st.session_state.local_parse_count
    cached = st.session_state.cache_hit_count
    total = api + local + cached
    print(f"[{now}] [{method}] [{CURRENT_USER}] \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
    if expense:
        print(f"         -> {expense.merchant} | {expense.amount} {expense.currency} | {expense.category}")
    print(f"         Session stats: {total} total parses | {api} API calls | {local} local | {cached} cache hits")
    print(f"         API cost ratio: {api}/{total} ({api/total*100:.0f}%)" if total > 0 else "")

def parse_expense_with_api(text: str) -> Expense | None:
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in st.session_state.parse_cache:
        st.session_state.cache_hit_count += 1
        expense = st.session_state.parse_cache[cache_key]
        _log_stats("CACHE HIT", text, expense)
        return expense

    prompt = f"""Extract expense info from this text as JSON.
Text: {text}
Today: {datetime.now().strftime('%Y-%m-%d')}
Return ONLY JSON: {{"date":"YYYY-MM-DD","merchant":"name","category":"Food|Transport|Shopping|Entertainment|Groceries|Utilities|Health|Other","currency":"HKD|TWD|USD|CNY|JPY|EUR|GBP|SGD|KRW|MYR","amount":0.0,"items":"description"}}"""

    try:
        result = llm.invoke(prompt)
        content = result.content.strip()
        if "```" in content:
            content = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL).group(1).strip()
        data = json.loads(content)
        expense = Expense(**data)
        st.session_state.parse_cache[cache_key] = expense
        st.session_state.api_call_count += 1
        _log_stats("API CALL", text, expense)
        return expense
    except Exception as e:
        st.session_state.api_call_count += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [API ERROR] [{CURRENT_USER}] \"{text[:60]}\" -> {str(e)}")
        st.error(f"Parsing failed: {str(e)}. Try clearer input or rephrase.")
        return None

def parse_expense_only(text: str):
    """Parse text into an Expense object (local first, then API fallback). Does NOT save to DB."""
    expense = try_local_parse(text)
    used_api = False

    if expense is not None:
        st.session_state.local_parse_count += 1
        _log_stats("LOCAL", text, expense)
    else:
        expense = parse_expense_with_api(text)
        used_api = True

    return expense, used_api

def save_expense(date, merchant, category, currency, amount, items, source):
    """Save a validated expense to the database."""
    amount_hkd = convert_to_hkd(amount, currency)
    conn.execute("""
        INSERT INTO expenses (username, date, merchant, category, currency, amount, amount_hkd, items, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (CURRENT_USER, date, merchant, category, currency, amount, amount_hkd, items, source))
    conn.commit()
    return amount_hkd

# ========================================
# Streamlit UI (main app — user is logged in)
# ========================================
st.title(f"🧾 {t('main_title')}")
st.write(t("main_desc"))

CATEGORIES = ["Food", "Transport", "Shopping", "Entertainment", "Groceries", "Utilities", "Health", "Other"]

# Tabs
tab_quick, tab_free, tab1, tab2 = st.tabs([
    f"⚡ {t('tab_quick')}", f"💬 {t('tab_free')}", f"📸 {t('tab_photo')}", f"🎤 {t('tab_voice')}"
])

# === Quick Form Tab ===
with tab_quick:
    with st.form("quick_form", clear_on_submit=True):
        qcol1, qcol2 = st.columns(2)
        with qcol1:
            q_date = st.date_input(t("quick_date"), value=datetime.now())
            q_merchant = st.text_input(t("quick_merchant"), placeholder="Starbucks")
            q_items = st.text_input(t("quick_items"), placeholder="Coffee, sandwich")
        with qcol2:
            q_currency = st.selectbox(t("quick_currency"), SUPPORTED_CURRENCIES, index=0)
            q_amount = st.number_input(t("quick_amount"), min_value=0.0, step=1.0, format="%.2f")
            q_category = st.selectbox(t("quick_category"), CATEGORIES)

        submitted = st.form_submit_button(f"💾 {t('quick_submit')}")
        if submitted and q_merchant and q_amount > 0:
            amount_hkd = convert_to_hkd(q_amount, q_currency)
            conn.execute("""
                INSERT INTO expenses (username, date, merchant, category, currency, amount, amount_hkd, items, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (CURRENT_USER, q_date.strftime('%Y-%m-%d'), q_merchant, q_category, q_currency,
                  q_amount, amount_hkd, q_items or q_merchant, "quick_form"))
            conn.commit()
            st.session_state.local_parse_count += 1
            _log_stats("FORM", f"{q_merchant} {q_amount} {q_currency}",
                       Expense(date=q_date.strftime('%Y-%m-%d'), merchant=q_merchant,
                               category=q_category, currency=q_currency, amount=q_amount, items=q_items or q_merchant))
            st.success(t("quick_success", merchant=q_merchant, amount=q_amount,
                         currency=q_currency, amount_hkd=amount_hkd,
                         category=q_category, date=q_date.strftime('%Y-%m-%d')))
            st.caption(t("quick_no_api"))

# === Free Text Tab ===
with tab_free:
    free_text = st.text_input(t("free_text_label"), key="free_text_input")

    if free_text.strip() and st.button(f"🔍 {t('free_text_btn')}"):
        partial = try_local_parse(free_text.strip())
        if partial:
            st.session_state.free_parsed = {
                "merchant": partial.merchant, "items": partial.items,
                "currency": partial.currency, "amount": partial.amount,
                "category": partial.category, "date": partial.date,
            }
        else:
            st.session_state.free_parsed = {
                "merchant": free_text.strip()[:30], "items": free_text.strip()[:50],
                "currency": "HKD", "amount": 0.0,
                "category": guess_category(free_text),
                "date": datetime.now().strftime('%Y-%m-%d'),
            }

    if "free_parsed" in st.session_state:
        p = st.session_state.free_parsed
        st.markdown("---")
        with st.form("free_text_form", clear_on_submit=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                f_merchant = st.text_input(t("quick_merchant"), value=p["merchant"])
                f_items = st.text_input(t("quick_items"), value=p["items"])
                f_date = st.date_input(t("quick_date"),
                                       value=datetime.strptime(p["date"], '%Y-%m-%d') if p["date"] else datetime.now())
            with fc2:
                f_currency = st.selectbox(t("quick_currency"), SUPPORTED_CURRENCIES,
                                          index=SUPPORTED_CURRENCIES.index(p["currency"]) if p["currency"] in SUPPORTED_CURRENCIES else 0)
                f_amount = st.number_input(t("quick_amount"), value=p["amount"], min_value=0.0, step=1.0, format="%.2f")
                f_category = st.selectbox(t("quick_category"), CATEGORIES,
                                          index=CATEGORIES.index(p["category"]) if p["category"] in CATEGORIES else len(CATEGORIES) - 1)

            if st.form_submit_button(f"💾 {t('free_text_confirm')}"):
                if f_merchant and f_amount > 0:
                    amount_hkd = convert_to_hkd(f_amount, f_currency)
                    conn.execute("""
                        INSERT INTO expenses (username, date, merchant, category, currency, amount, amount_hkd, items, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (CURRENT_USER, f_date.strftime('%Y-%m-%d'), f_merchant, f_category, f_currency,
                          f_amount, amount_hkd, f_items or f_merchant, "free_text"))
                    conn.commit()
                    st.session_state.local_parse_count += 1
                    _log_stats("FREE TEXT", f"{f_merchant} {f_amount} {f_currency}",
                               Expense(date=f_date.strftime('%Y-%m-%d'), merchant=f_merchant,
                                       category=f_category, currency=f_currency, amount=f_amount, items=f_items or f_merchant))
                    st.success(t("free_text_saved"))
                    st.caption(t("quick_no_api"))
                    del st.session_state.free_parsed

# === Photo Receipt Tab ===
with tab1:
    uploaded_file = st.file_uploader(t("upload_label"), type=['png', 'jpg', 'jpeg', 'pdf'])
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        is_pdf = uploaded_file.name.lower().endswith('.pdf')

        with st.spinner(t("spinner_ocr")):
            if is_pdf:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                all_page_texts = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    result = reader.readtext(img_bytes, detail=0, paragraph=True)
                    page_text = "\n".join(result)
                    if page_text.strip():
                        all_page_texts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                doc.close()
                extracted_text = "\n\n".join(all_page_texts) if all_page_texts else ""
                if not extracted_text.strip():
                    st.warning("No text found in the PDF. It may be a scanned image or empty.")
            else:
                result = reader.readtext(file_bytes, detail=0, paragraph=True)
                extracted_text = "\n".join(result)

        st.write(t("extracted_text"))
        st.code(extracted_text)

        # Step 1: Parse — store result for review
        if st.button(f"🧠 {t('btn_parse_review')}"):
            with st.spinner(t("spinner_ai")):
                expense, used_api = parse_expense_only(extracted_text)
                if expense:
                    st.session_state.photo_parsed = {
                        "merchant": expense.merchant, "items": expense.items,
                        "currency": expense.currency, "amount": expense.amount,
                        "category": expense.category, "date": expense.date,
                    }
                    st.session_state.photo_used_api = used_api
                else:
                    st.session_state.photo_parsed = {
                        "merchant": "", "items": extracted_text[:50],
                        "currency": "HKD", "amount": 0.0,
                        "category": "Other", "date": datetime.now().strftime('%Y-%m-%d'),
                    }
                    st.session_state.photo_used_api = used_api

        # Step 2: Editable review form
        if "photo_parsed" in st.session_state:
            p = st.session_state.photo_parsed
            st.info(t("review_parsed_local") if not st.session_state.get("photo_used_api") else t("review_parsed_api"))
            with st.form("photo_review_form", clear_on_submit=True):
                st.markdown(f"**{t('review_header')}**")
                pc1, pc2 = st.columns(2)
                with pc1:
                    p_merchant = st.text_input(t("quick_merchant"), value=p["merchant"])
                    p_items = st.text_input(t("quick_items"), value=p["items"])
                    p_date = st.date_input(t("quick_date"),
                                           value=datetime.strptime(p["date"], '%Y-%m-%d') if p["date"] else datetime.now())
                with pc2:
                    p_currency = st.selectbox(t("quick_currency"), SUPPORTED_CURRENCIES,
                                              index=SUPPORTED_CURRENCIES.index(p["currency"]) if p["currency"] in SUPPORTED_CURRENCIES else 0)
                    p_amount = st.number_input(t("quick_amount"), value=p["amount"], min_value=0.0, step=1.0, format="%.2f")
                    p_category = st.selectbox(t("quick_category"), CATEGORIES,
                                              index=CATEGORIES.index(p["category"]) if p["category"] in CATEGORIES else len(CATEGORIES) - 1)

                if st.form_submit_button(f"💾 {t('review_save')}"):
                    if p_merchant and p_amount > 0:
                        amount_hkd = save_expense(p_date.strftime('%Y-%m-%d'), p_merchant, p_category,
                                                  p_currency, p_amount, p_items or p_merchant, "receipt_photo")
                        _log_stats("PHOTO", f"{p_merchant} {p_amount} {p_currency}",
                                   Expense(date=p_date.strftime('%Y-%m-%d'), merchant=p_merchant,
                                           category=p_category, currency=p_currency, amount=p_amount, items=p_items or p_merchant))
                        st.success(t("success_added", merchant=p_merchant, amount=p_amount,
                                     currency=p_currency, amount_hkd=amount_hkd,
                                     category=p_category, date=p_date.strftime('%Y-%m-%d')))
                        del st.session_state.photo_parsed

# === Voice Input Tab ===
with tab2:
    audio_bytes = st.audio_input(t("voice_label"))
    if audio_bytes:
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes.getvalue())

        with st.spinner(t("spinner_transcribe")):
            result = whisper_model.transcribe(temp_path)
            transcribed_text = result["text"].strip()

        st.write(t("transcribed_text"))
        st.code(transcribed_text)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Step 1: Parse — store result for review
        if st.button(f"🧠 {t('btn_parse_voice_review')}"):
            with st.spinner(t("spinner_ai")):
                expense, used_api = parse_expense_only(transcribed_text)
                if expense:
                    st.session_state.voice_parsed = {
                        "merchant": expense.merchant, "items": expense.items,
                        "currency": expense.currency, "amount": expense.amount,
                        "category": expense.category, "date": expense.date,
                    }
                    st.session_state.voice_used_api = used_api
                else:
                    st.session_state.voice_parsed = {
                        "merchant": "", "items": transcribed_text[:50],
                        "currency": "HKD", "amount": 0.0,
                        "category": "Other", "date": datetime.now().strftime('%Y-%m-%d'),
                    }
                    st.session_state.voice_used_api = used_api

        # Step 2: Editable review form
        if "voice_parsed" in st.session_state:
            vp = st.session_state.voice_parsed
            st.info(t("review_parsed_local") if not st.session_state.get("voice_used_api") else t("review_parsed_api"))
            with st.form("voice_review_form", clear_on_submit=True):
                st.markdown(f"**{t('review_header')}**")
                vc1, vc2 = st.columns(2)
                with vc1:
                    v_merchant = st.text_input(t("quick_merchant"), value=vp["merchant"])
                    v_items = st.text_input(t("quick_items"), value=vp["items"])
                    v_date = st.date_input(t("quick_date"),
                                           value=datetime.strptime(vp["date"], '%Y-%m-%d') if vp["date"] else datetime.now())
                with vc2:
                    v_currency = st.selectbox(t("quick_currency"), SUPPORTED_CURRENCIES,
                                              index=SUPPORTED_CURRENCIES.index(vp["currency"]) if vp["currency"] in SUPPORTED_CURRENCIES else 0)
                    v_amount = st.number_input(t("quick_amount"), value=vp["amount"], min_value=0.0, step=1.0, format="%.2f")
                    v_category = st.selectbox(t("quick_category"), CATEGORIES,
                                              index=CATEGORIES.index(vp["category"]) if vp["category"] in CATEGORIES else len(CATEGORIES) - 1)

                if st.form_submit_button(f"💾 {t('review_save')}"):
                    if v_merchant and v_amount > 0:
                        amount_hkd = save_expense(v_date.strftime('%Y-%m-%d'), v_merchant, v_category,
                                                  v_currency, v_amount, v_items or v_merchant, "voice")
                        _log_stats("VOICE", f"{v_merchant} {v_amount} {v_currency}",
                                   Expense(date=v_date.strftime('%Y-%m-%d'), merchant=v_merchant,
                                           category=v_category, currency=v_currency, amount=v_amount, items=v_items or v_merchant))
                        st.success(t("success_added", merchant=v_merchant, amount=v_amount,
                                     currency=v_currency, amount_hkd=amount_hkd,
                                     category=v_category, date=v_date.strftime('%Y-%m-%d')))
                        del st.session_state.voice_parsed

# ========================================
# Display All Expenses (filtered by current user)
# ========================================
st.divider()
st.header(f"📊 {t('header_all_expenses')}")

raw_df = pd.read_sql_query(
    "SELECT id, date, merchant, category, currency, amount, amount_hkd, items, source FROM expenses WHERE username = ? ORDER BY date DESC",
    conn, params=(CURRENT_USER,)
)

if not raw_df.empty:
    display_df = raw_df.copy()
    # Add a checkbox column for delete selection
    display_df.insert(0, t('col_select'), False)

    # Editable table — users can edit date, merchant, category, currency, amount, items directly
    edited_df = st.data_editor(
        display_df.drop(columns=['id']),
        use_container_width=True,
        hide_index=True,
        disabled=['amount_hkd', 'source'],
        column_config={
            t('col_select'): st.column_config.CheckboxColumn(t('col_select'), default=False),
            'date': st.column_config.TextColumn('date'),
            'category': st.column_config.SelectboxColumn('category', options=CATEGORIES),
            'currency': st.column_config.SelectboxColumn('currency', options=SUPPORTED_CURRENCIES),
            'amount': st.column_config.NumberColumn('amount', min_value=0.0, step=1.0, format="%.2f"),
            'amount_hkd': st.column_config.NumberColumn('amount_hkd', format="%.2f"),
        },
        key="expense_editor",
    )

    # Action buttons side by side
    btn_col1, btn_col2 = st.columns(2)

    # Save Changes button
    with btn_col1:
        if st.button(f"💾 {t('save_changes')}", type="primary"):
            update_count = 0
            for idx in range(len(edited_df)):
                row_id = raw_df.iloc[idx]['id']
                orig = raw_df.iloc[idx]
                ed = edited_df.iloc[idx]
                # Check if any editable field changed
                changed = (
                    str(ed['date']) != str(orig['date']) or
                    str(ed['merchant']) != str(orig['merchant']) or
                    str(ed['category']) != str(orig['category']) or
                    str(ed['currency']) != str(orig['currency']) or
                    float(ed['amount']) != float(orig['amount']) or
                    str(ed['items']) != str(orig['items'])
                )
                if changed:
                    new_amount_hkd = convert_to_hkd(float(ed['amount']), ed['currency'])
                    conn.execute("""
                        UPDATE expenses SET date=?, merchant=?, category=?, currency=?, amount=?, amount_hkd=?, items=?
                        WHERE id=? AND username=?
                    """, (ed['date'], ed['merchant'], ed['category'], ed['currency'],
                          float(ed['amount']), new_amount_hkd, ed['items'], int(row_id), CURRENT_USER))
                    update_count += 1
            if update_count > 0:
                conn.commit()
                st.success(t("save_changes_success", count=update_count))
                st.rerun()
            else:
                st.info(t("save_changes_none"))

    # Delete Selected button
    with btn_col2:
        if st.button(f"🗑️ {t('delete_selected')}", type="secondary"):
            selected_mask = edited_df[t('col_select')] == True
            if selected_mask.any():
                ids_to_delete = raw_df.loc[selected_mask.values, 'id'].tolist()
                placeholders = ','.join('?' * len(ids_to_delete))
                conn.execute(f"DELETE FROM expenses WHERE id IN ({placeholders}) AND username = ?",
                             ids_to_delete + [CURRENT_USER])
                conn.commit()
                st.success(t("delete_success", count=len(ids_to_delete)))
                st.rerun()
            else:
                st.warning(t("delete_none"))

    # =============================
    # Monthly Summary Dashboard
    # =============================
    st.divider()
    st.header(f"📈 {t('header_monthly')}")

    raw_df['date_parsed'] = pd.to_datetime(raw_df['date'], errors='coerce')
    raw_df['year_month'] = raw_df['date_parsed'].dt.strftime('%Y-%m')
    available_months = sorted(raw_df['year_month'].dropna().unique(), reverse=True)
    current_month = datetime.now().strftime('%Y-%m')

    selected_month = st.selectbox(
        t("select_month"),
        available_months,
        index=available_months.index(current_month) if current_month in available_months else 0,
    )

    month_df = raw_df[raw_df['year_month'] == selected_month].copy()
    month_label = datetime.strptime(selected_month, '%Y-%m').strftime('%B %Y')

    if not month_df.empty:
        total_hkd = month_df['amount_hkd'].sum()
        num_transactions = len(month_df)
        avg_per_transaction = total_hkd / num_transactions if num_transactions > 0 else 0
        num_days = month_df['date_parsed'].dt.date.nunique()
        avg_per_day = total_hkd / num_days if num_days > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("metric_total"), f"${total_hkd:,.2f}")
        col2.metric(t("metric_transactions"), f"{num_transactions}")
        col3.metric(t("metric_avg_txn"), f"${avg_per_transaction:,.2f}")
        col4.metric(t("metric_avg_day"), f"${avg_per_day:,.2f}")

        st.subheader(t("sub_category", month=month_label))
        cat_df = month_df.groupby('category')['amount_hkd'].sum().sort_values(ascending=True).reset_index()
        cat_df.columns = [t('col_category'), t('col_amount_hkd')]
        st.bar_chart(cat_df, x=t('col_category'), y=t('col_amount_hkd'), horizontal=True)

        st.subheader(t("sub_daily", month=month_label))
        daily_df = month_df.groupby(month_df['date_parsed'].dt.date)['amount_hkd'].sum().reset_index()
        daily_df.columns = ['Date', t('col_amount_hkd')]
        daily_df = daily_df.sort_values('Date')
        st.line_chart(daily_df, x='Date', y=t('col_amount_hkd'))

        st.subheader(t("sub_merchants", month=month_label))
        merch_df = month_df.groupby('merchant')['amount_hkd'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10).reset_index()
        merch_df.columns = [t('col_merchant'), t('col_total_hkd'), t('col_visits')]
        merch_df[t('col_total_hkd')] = merch_df[t('col_total_hkd')].apply(lambda x: f"${x:,.2f}")
        st.dataframe(merch_df, use_container_width=True, hide_index=True)

        cur_df = month_df.groupby('currency')['amount_hkd'].sum().sort_values(ascending=False).reset_index()
        cur_df.columns = [t('col_currency'), t('col_total_hkd')]
        if len(cur_df) > 1:
            st.subheader(t("sub_currency", month=month_label))
            st.bar_chart(cur_df, x=t('col_currency'), y=t('col_total_hkd'))
    else:
        st.info(t("no_expenses_month", month=month_label))
else:
    st.info(t("no_expenses_yet"))

# Footer
st.caption(t("footer"))
