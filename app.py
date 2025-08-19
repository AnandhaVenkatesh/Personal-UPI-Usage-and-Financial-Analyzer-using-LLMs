import os
import io
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
from rapidfuzz import process, fuzz

# ---------- Optional OCR (graceful if not present) ----------
# have to local installs, you can set these:
#   e.g., r"D:\UPI_Project\Tesseract-OCR\tesseract.exe"
#   e.g., r"D:\UPI_Project\bin\poppler-24.08.0\bin"
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "").strip()
POPPLER_PATH   = os.getenv("POPPLER_PATH", "").strip()

OCR_AVAILABLE = False
try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_bytes  # type: ignore
    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- LLM (Gemini) setup ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_READY = False
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GENAI_READY = True
except Exception:
    GENAI_READY = False

# ---------- Streamlit page config / style ----------
st.set_page_config(page_title="AI Personal Finance Assistant", page_icon="💰", layout="wide")
st.markdown(
    """
    <style>
    .main-title { text-align:center; font-size:34px; font-weight:700; color:#4CAF50; text-shadow: 2px 2px 5px rgba(76,175,80,.4);} 
    .sub-title { text-align:center; font-size:18px; color:#ddd; margin-bottom:20px;}
    .result-card { background: rgba(0, 150, 136, 0.08); padding: 16px; border-radius: 10px; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,.06);} 
    .success-banner { background: linear-gradient(to right, #2E7D32, #1B5E20); color:#fff; padding:14px; border-radius:10px; text-align:center; font-weight:700; margin-top:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<h1 class="main-title">💰 AI-Powered Personal Finance Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload your UPI/Bank statements (Paytm, GPay, PhonePe, etc.) → Parse → Analyze → Get insights</p>', unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("ℹ️ How to Use")
st.sidebar.write("1) Upload one or more statement PDFs.")
st.sidebar.write("2) Review parsed transactions.")
st.sidebar.write("3) (Optional) Enter monthly income for savings%.")
st.sidebar.write("4) Generate LLM insights or use offline analytics.")

with st.sidebar.expander("⚙️ Settings", expanded=False):
    user_monthly_income = st.number_input("Monthly Income (₹) — optional", min_value=0, step=1000, value=0)
    enable_ocr = st.checkbox("Enable OCR fallback (slower)", value=False)
    chunk_limit = st.number_input("Max rows sent to LLM", min_value=100, max_value=5000, value=800, step=100)
    show_offline_insights = st.checkbox("Show offline insights (no LLM)", value=True)

if GEMINI_API_KEY:
    st.sidebar.success("Gemini API key detected from environment.")
else:
    st.sidebar.warning("No GEMINI_API_KEY found. LLM features will be disabled. Offline insights still work.")

# ---------- Constants / Regex ----------
DATE_PATTERNS = [
    # dd-first
    "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y", "%d %B %Y",
    # month-first (alphabetical)
    "%b %d, %Y", "%B %d, %Y", "%b %d,%Y", "%B %d,%Y"
]

ALPHA_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
DATE_RE = re.compile(
    rf"\b(\d{{2}}[/-]\d{{2}}[/-]\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{2}}\s?[A-Za-z]{{3}}\s?\d{{4}}|{ALPHA_MONTHS}[a-z]*\s+\d{{1,2}},\s*\d{{4}})\b",
    re.I,
)

# Prefer rupee-anchored amount for Indian statements; keep a generic one too
RUPEE_AMOUNT_RE = re.compile(r"[₹₹]\s*([0-9][0-9,]*(?:\.\d{1,2})?)")
AMOUNT_RE = re.compile(r"(?:₹|INR)?\s*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})|[+-]?\d+(?:\.\d{1,2}))")

DEBIT_CREDIT_RE = re.compile(r"\b(DEBIT|CREDIT|DR|CR)\b", re.I)
TXN_ID_RE = re.compile(r"(?:Transaction\s*ID|Txn\s*ID)\s*[:\-]?\s*([A-Za-z0-9\-_]+)", re.I)
UTR_RE = re.compile(r"\bUTR\s*(?:No\.?|ID)?\s*[:\-]?\s*([A-Za-z0-9]+)\b", re.I)
PAID_TO_RE = re.compile(r"(?:Paid\s+to|Received\s+from)\s+(.+?)(?:\s+(?:DEBIT|CREDIT|DR|CR)\b|\s+₹|$)", re.I)
TIME_RE = re.compile(r"\b((?:[01]?\d|2[0-3]):[0-5]\d(?:\s?(?:AM|PM))?)\b", re.I)

DEBIT_KEYS = {
    "debit", "debited", "withdrawal", "withdraw", "purchase", "payment", "paid", "upi to",
    "money sent", "sent", "transfer to", "atm", "pos", "bill", "emi", "petrol", "fuel", "dining",
    "zomato", "swiggy", "insurance premium", "online purchase"
}
CREDIT_KEYS = {
    "credit", "credited", "cr", "received", "refund", "reversal", "cashback", "salary",
    "deposit", "transfer from", "interest", "redeem", "redemption"
}
BALANCE_KEYS = {"opening balance", "closing balance", "balance only", "closingbal", "openingbal"}

CATEGORY_RULES = {
    "food": ["swiggy", "zomato", "restaurant", "hotel", "food"],
    "grocery": ["bigbasket", "dmart", "grocery", "mart", "supermarket"],
    "utilities": ["electricity", "water", "gas", "biller", "broadband", "internet", "dth"],
    "transport": ["uber", "ola", "metro", "fuel", "petrol", "diesel", "fastag"],
    "shopping": ["amazon", "flipkart", "myntra", "ajio", "store", "shop"],
    "entertainment": ["netflix", "prime", "hotstar", "movie", "bookmyshow"],
    "financial": ["bank", "emi", "loan", "insurance", "sip", "mutual fund"],
    "transfer": ["upi", "imps", "neft", "rtgs", "to", "from"],
}

UPI_ID_RE = re.compile(r"[A-Za-z0-9._-]+@[A-Za-z]+")
MERCHANT_CANON = {
    "swiggy": "Swiggy", "zomato": "Zomato", "amazon": "Amazon",
    "flipkart": "Flipkart", "uber": "Uber", "ola": "Ola"
}

# ---------- Helpers ----------
def _normalize_date(s: str) -> str | None:
    s = re.sub(r"\s+", " ", (s or "").strip())
    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None

def _infer_method_from_source(name: str) -> str:
    low = (name or "").lower()
    if "paytm" in low:
        return "Paytm"
    if "phonepe" in low:
        return "PhonePe"
    if "gpay" in low or "google" in low:
        return "GPay"
    return "UPI"

def _clean_description(desc: str) -> str:
    """Remove leading date/time from Paytm-style descriptions."""
    if not desc:
        return ""
    return re.sub(r"^\d{2}[-/]\d{2}[-/]\d{4}\s+\d{1,2}:\d{2}", "", desc).strip()

def _canonical_merchant(name: str) -> str:
    low = (name or "").lower()
    for key, canon in MERCHANT_CANON.items():
        if key in low:
            return canon
    try:
        best = process.extractOne(low, list(MERCHANT_CANON.keys()), scorer=fuzz.WRatio)
        if best and isinstance(best, tuple) and len(best) >= 2 and best[1] > 85:
            return MERCHANT_CANON.get(best[0], name)
    except Exception:
        pass
    return (name or "").strip()[:80]

def _categorize(desc: str) -> str:
    d = (desc or "").lower()
    for cat, kws in CATEGORY_RULES.items():
        if any(k in d for k in kws):
            return cat
    return "other"

def _to_number_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def _find_amounts(txt: str) -> list[float]:
    vals: list[float] = []
    for m in AMOUNT_RE.findall(txt or ""):
        s = m if isinstance(m, str) else (m[0] if isinstance(m, (list, tuple)) else str(m))
        try:
            vals.append(float(s.replace(",", "")))
        except Exception:
            pass
    return vals

def _guess_type_from_text(txt: str) -> str:
    low = (txt or "").lower()
    if any(k in low for k in BALANCE_KEYS):
        return "balance"
    if re.search(r"\bcr\b", low):
        return "credit"
    if re.search(r"\bdr\b", low):
        return "debit"
    if any(k in low for k in CREDIT_KEYS):
        return "credit"
    if any(k in low for k in DEBIT_KEYS):
        return "debit"
    if "transfer from" in low or ("from" in low and "transfer" in low):
        return "credit"
    if "transfer to" in low or ("to" in low and "transfer" in low):
        return "debit"
    return "unknown"

def _extract_amount_and_type(txt: str) -> tuple[float | None, str, float | None]:
    # Prefer ₹-anchored amount when present
    mru = RUPEE_AMOUNT_RE.search(txt or "")
    tx_amt = float(mru.group(1).replace(",", "")) if mru else None

    tkn = DEBIT_CREDIT_RE.search(txt or "")
    if tkn:
        token = tkn.group(1).upper()
        tx_type = "debit" if token in ("DEBIT", "DR") else "credit"
    else:
        tx_type = _guess_type_from_text(txt)

    balance_amt = None
    if not mru:
        amounts = _find_amounts(txt)
        if len(amounts) >= 2:
            tx_amt, balance_amt = amounts[0], amounts[-1]
        elif len(amounts) == 1:
            tx_amt = amounts[0]

    return (tx_amt, tx_type, balance_amt)

# ---------- PDF text extraction (with OCR fallback) ----------
def _extract_texts_from_pdf(file_bytes: bytes, use_ocr: bool = False) -> str:
    text_chunks: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            trigger_ocr = (len((t or "").strip()) < 30)
            if trigger_ocr and use_ocr and OCR_AVAILABLE:
                try:
                    kwargs = {"first_page": page.page_number, "last_page": page.page_number}
                    if POPPLER_PATH:
                        kwargs["poppler_path"] = POPPLER_PATH
                    images = convert_from_bytes(file_bytes, **kwargs)
                    if images:
                        t = pytesseract.image_to_string(images[0]) or ""
                except Exception:
                    t = t or ""  # keep whatever we had
            text_chunks.append(t)
    return "\n".join(text_chunks).strip()

# ---------- Table extraction ----------
def _extract_table_from_pdf(file_bytes: bytes) -> pd.DataFrame:
    all_tables = []
    headers = None
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                if not headers:
                    headers = table[0]
                all_tables.extend(table[1:])
            else:
                text = page.extract_text() or ""
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if not headers and any("date" in ln.lower() for ln in lines):
                    headers = re.split(r"\s{2,}", lines[0])
                    lines = lines[1:]
                for ln in lines:
                    parts = re.split(r"\s{2,}", ln)
                    if len(parts) >= 3:
                        all_tables.append(parts)
    if headers and all_tables:
        return pd.DataFrame(all_tables, columns=headers[:len(all_tables[0])])
    elif all_tables:
        return pd.DataFrame(all_tables, columns=[f"Col{i+1}" for i in range(len(all_tables[0]))])
    return pd.DataFrame()

def _standardize_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if "date" in lc:
            mapping[c] = "Date"
        elif "time" in lc:
            mapping[c] = "Time"
        elif any(k in lc for k in ["desc", "narrat", "merchant"]):
            mapping[c] = "Description"
        elif "type" in lc:
            mapping[c] = "Type"
        elif "debit" in lc:
            mapping[c] = "Debit"
        elif "credit" in lc:
            mapping[c] = "Credit"
        elif "amount" in lc:
            mapping[c] = "Amount"
        elif "balance" in lc:
            mapping[c] = "Balance"
        elif "status" in lc:
            mapping[c] = "Status"
        elif "transaction id" in lc or re.search(r"\btxn\b", lc or ""):
            mapping[c] = "TransactionID"
        elif "utr" in lc:
            mapping[c] = "UTR"
    return df.rename(columns=mapping)

def _repair_bank_table(df: pd.DataFrame) -> pd.DataFrame:
    if "Description" in df.columns and ("Debit" not in df.columns and "Credit" not in df.columns):
        fixed_rows = []
        for val in df["Description"]:
            parts = re.split(r"\s{2,}", str(val).strip())
            if len(parts) >= 4:
                date, desc, amt, bal = parts[0], " ".join(parts[1:-2]), parts[-2], parts[-1]
                debit, credit = ("", amt) if "-" not in amt else (amt.replace("-", ""), "")
                fixed_rows.append([date, desc, debit, credit, bal])
        if fixed_rows:
            return pd.DataFrame(fixed_rows, columns=["Date", "Description", "Debit", "Credit", "Balance"])
    return df

def _extract_tables_from_pdf(file_bytes: bytes) -> list[dict]:
    rows = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            try:
                table = page.extract_table()
                if not table:
                    continue
                headers = [h.strip().lower() if h else "" for h in table[0]]
                for r in table[1:]:
                    if not r or len(r) < 3:
                        continue
                    row_dict = {headers[i]: (r[i] or "").strip() for i in range(len(r))}
                    date_str = row_dict.get("date", "") or row_dict.get("txn date", "")
                    time_str = row_dict.get("time", "")
                    desc = row_dict.get("description", "") or row_dict.get("narration", "")
                    clean_desc = _clean_description(desc)
                    tx_type = (row_dict.get("type", "") or row_dict.get("dr/cr", "")).lower()
                    amt_raw = row_dict.get("amount", "") or row_dict.get("amt", "")
                    amt_raw = amt_raw.replace(",", "").replace("₹", "").strip()
                    try:
                        amt = float(amt_raw) if amt_raw else None
                    except Exception:
                        amt = None
                    if amt is None:
                        # Bank style: separate debit/credit columns
                        d = row_dict.get("debit", "").replace(",", "").replace("₹", "").strip()
                        c = row_dict.get("credit", "").replace(",", "").replace("₹", "").strip()
                        if d:
                            amt = float(d); tx_type = "debit"
                        elif c:
                            amt = float(c); tx_type = "credit"
                    bal_raw = row_dict.get("balance", "") or row_dict.get("bal", "")
                    try:
                        bal = float(bal_raw.replace(",", "").replace("₹", "")) if bal_raw else None
                    except Exception:
                        bal = None

                    if amt is not None:
                        rows.append({
                            "date": _normalize_date(date_str) or "",
                            "time": time_str,
                            "amount": float(amt),
                            "type": "credit" if ("cr" in tx_type or "credit" in tx_type) else ("debit" if ("dr" in tx_type or "debit" in tx_type) else "unknown"),
                            "merchant": _canonical_merchant(desc),
                            "description": desc,
                            "method": "Paytm" if ("paytm" in "".join(headers) or "paytm" in desc.lower()) else "UPI",
                            "source_pdf": "table_extract",
                            "running_balance": bal if bal is not None else np.nan,
                            "bank_txn_id": row_dict.get("transaction id", "") or row_dict.get("txn id", ""),
                            "utr": row_dict.get("utr", ""),
                        })
            except Exception:
                continue
    return rows

# ---------- Text parsing ----------
def _split_into_transactions(page_text: str) -> list[str]:
    pattern = re.compile(rf"(?=(?:{DATE_RE.pattern}))", re.I)
    parts = pattern.split(page_text or "")
    blocks = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        if re.search(r"\b(DEBIT|CREDIT|DR|CR)\b", p, re.I) or "₹" in p or "INR" in p.upper():
            blocks.append(p)
    return blocks

def _parse_block(txt: str, source_pdf: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not txt:
        return rows

    date_match = DATE_RE.search(txt)
    date_str = _normalize_date(date_match.group(0)) if date_match else ""
    time_match = TIME_RE.search(txt)
    time_str = time_match.group(0).strip() if time_match else ""

    tx_amt, tx_type, balance_amt = _extract_amount_and_type(txt)

    # merchant / description
    merch = ""
    m = PAID_TO_RE.search(txt)
    if m:
        merch = m.group(1).strip()
    if not merch:
        upi_match = UPI_ID_RE.search(txt or "")
        merch = upi_match.group(0) if upi_match else ""
    if not merch:
        tokens = [t for t in re.split(r"[^A-Za-z0-9@._-]+", txt or "") if t and not t.replace('.', '', 1).isdigit()]
        merch = max(tokens, key=len)[:80] if tokens else "Unknown"

    description = " ".join(re.split(r"\s+", txt)).strip()
    description = _clean_description(description)
    if "opening balance" in description.lower() or "closing balance" in description.lower():
        return rows
    txn_id = (TXN_ID_RE.search(txt).group(1).strip() if TXN_ID_RE.search(txt) else "")
    utr = (UTR_RE.search(txt).group(1).strip() if UTR_RE.search(txt) else "")

    if tx_amt is None:
        return rows

    rows.append({
        "date": date_str,
        "time": time_str,
        "amount": float(tx_amt),
        "type": tx_type if tx_type in {"debit", "credit"} else "unknown",
        "merchant": _canonical_merchant(merch or "Unknown"),
        "description": description[:300],
        "method": _infer_method_from_source(source_pdf),
        "source_pdf": source_pdf,
        "running_balance": balance_amt if balance_amt is not None else np.nan,
        "bank_txn_id": txn_id,
        "utr": utr,
    })
    return rows

# ---------- Combined extraction pipeline ----------
@st.cache_data(show_spinner=False)
def extract_and_parse(files: List[Tuple[str, bytes]], use_ocr: bool = False) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []

    for filename, file_bytes in files:
        # 1) Try structured tables first
        table_df = _extract_table_from_pdf(file_bytes)
        if not table_df.empty:
            table_df = _standardize_table_columns(table_df)
            table_df = _repair_bank_table(table_df)

            # Case A: Type + Amount
            if ("Type" in table_df.columns) and ("Amount" in table_df.columns):
                table_df["Amount"] = _to_number_series(table_df["Amount"])
                for _, row in table_df.iterrows():
                    date_str = _normalize_date(str(row.get("Date", ""))) or ""
                    time_str = str(row.get("Time", "")).strip()
                    desc     = str(row.get("Description", "")).strip()
                        # 🚫 Skip Opening/Closing Balance rows
                    if "opening balance" in desc.lower() or "closing balance" in desc.lower():
                        continue
                    raw_type = str(row.get("Type", "")).strip().lower()
                    amount   = row.get("Amount", np.nan)

                    if pd.notna(amount):
                        tx_type = "credit" if ("credit" in raw_type or "cr" in raw_type) else (
                                  "debit"  if ("debit"  in raw_type or "dr" in raw_type)  else "unknown")

                        all_rows.append({
                            "date": date_str,
                            "time": time_str,
                            "amount": float(amount),
                            "type": tx_type,
                            "merchant": _canonical_merchant(desc),
                            "description": desc,
                            "method": _infer_method_from_source(filename),
                            "source_pdf": filename,
                            "running_balance": np.nan,
                            "bank_txn_id": str(row.get("TransactionID", "") or "").strip(),
                            "utr": str(row.get("UTR", "") or "").strip(),
                        })

            # Case B: Bank style Debit/Credit columns
            elif any(c in table_df.columns for c in ["Debit", "Credit"]):
                for _, row in table_df.iterrows():
                    date_str = _normalize_date(str(row.get("Date", ""))) or ""
                    desc     = str(row.get("Description", "")).strip()
                    debit    = _to_number_series(pd.Series([row.get("Debit", "")])).iloc[0]
                    credit   = _to_number_series(pd.Series([row.get("Credit", "")])).iloc[0]
                    balance  = _to_number_series(pd.Series([row.get("Balance", "")])).iloc[0]

                    d = float(debit) if pd.notna(debit) else 0.0
                    c = float(credit) if pd.notna(credit) else 0.0

                    if d > 0:
                        all_rows.append({
                            "date": date_str, "time": "", "amount": d, "type": "debit",
                            "merchant": _canonical_merchant(desc), "description": desc,
                            "method": _infer_method_from_source(filename), "source_pdf": filename,
                            "running_balance": float(balance) if pd.notna(balance) else np.nan,
                            "bank_txn_id": str(row.get("TransactionID", "") or "").strip(),
                            "utr": str(row.get("UTR", "") or "").strip(),
                        })
                    if c > 0:
                        all_rows.append({
                            "date": date_str, "time": "", "amount": c, "type": "credit",
                            "merchant": _canonical_merchant(desc), "description": desc,
                            "method": _infer_method_from_source(filename), "source_pdf": filename,
                            "running_balance": float(balance) if pd.notna(balance) else np.nan,
                            "bank_txn_id": str(row.get("TransactionID", "") or "").strip(),
                            "utr": str(row.get("UTR", "") or "").strip(),
                        })

            # Additionally parse row-wise tables to catch missed meta
            all_rows.extend(_extract_tables_from_pdf(file_bytes))

        # 2) Text/regex fallback (handles Paytm SMS-like / "tns" style too)
        raw_text = _extract_texts_from_pdf(file_bytes, use_ocr=use_ocr)
        tx_blocks = _split_into_transactions(raw_text)
        if not tx_blocks:
            # line-accumulation fallback: detect date starts
            lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
            block: List[str] = []
            for ln in lines:
                if DATE_RE.search(ln) and block:
                    tx_blocks.append(" ".join(block)); block = [ln]
                else:
                    block.append(ln)
            if block:
                tx_blocks.append(" ".join(block))

        for blk in tx_blocks:
            rows = _parse_block(blk, filename)
            all_rows.extend(rows)

        # ✅ Deduplicate if both table + text parsing added rows
    unique_rows = {}
    for row in all_rows:
        # A transaction is unique by date + amount + description
        key = (row.get("date"), row.get("amount"), row.get("description"))
        unique_rows[key] = row
    all_rows = list(unique_rows.values())


    # 3) Post-process
    df = pd.DataFrame(all_rows).drop_duplicates()
    if df.empty:
        return df

    df["date"] = df["date"].apply(lambda x: _normalize_date(str(x)) or "")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"]).reset_index(drop=True)

    df["row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["month"] = df["date"].apply(lambda s: s[:7] if isinstance(s, str) and len(s) >= 7 else "")
    df["merchant"] = df["merchant"].fillna("").apply(_canonical_merchant)
    # if "merchant" in df.columns:
    #     df = df.drop(columns=["merchant"])
    df["category"] = df["description"].fillna("").apply(_categorize)

    # Normalize type
    df["type"] = df["type"].fillna("")
    df.loc[~df["type"].isin(["debit", "credit"]), "type"] = "unknown"

    # Currency heuristic per row
    def _currency_heur(row):
        txt = f"{row.get('description','')} {row.get('method','')} {row.get('source_pdf','')}"
        return "INR" if ("₹" in txt or row.get("method") in ["Paytm", "PhonePe", "GPay", "UPI"]) else "USD"
    df["currency"] = df.apply(_currency_heur, axis=1)

    # Prefer bank_txn_id when present; else row_id
    if "bank_txn_id" in df.columns:
        txid = df["bank_txn_id"].replace({"": np.nan})
        df["transaction_id"] = txid.fillna(df["row_id"])
    else:
        df["transaction_id"] = df["row_id"]

    df["signed_amount"] = np.where(
        df["type"] == "credit", df["amount"],
        np.where(df["type"] == "debit", -df["amount"], np.nan)
    )

    # Ensure consistent columns
    for col in ["utr", "running_balance", "time", "method", "source_pdf"]:
        if col not in df.columns:
            df[col] = np.nan

    return df

# ---------- Offline analytics ----------
def compute_offline_insights(df: pd.DataFrame, income: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"months": {}, "top_categories": [], "top_merchants": [], "notes": []}
    if df.empty:
        return out
    known = df[df["type"].isin(["debit", "credit"])].copy()
    month_grp = known.groupby(["month", "type"]).agg(total=("amount", "sum")).reset_index()
    months: Dict[str, Dict[str, float]] = {}
    for _, r in month_grp.iterrows():
        m = r["month"] or "unknown"
        if m not in months:
            months[m] = {"income": 0.0, "expenses": 0.0}
        if r["type"] == "credit":
            months[m]["income"] += float(r["total"])
        elif r["type"] == "debit":
            months[m]["expenses"] += float(r["total"])
    out["months"] = months
    cat = known[known["type"] == "debit"].groupby("category")["amount"].sum().sort_values(ascending=False)
    out["top_categories"] = [(k, float(v)) for k, v in cat.head(10).items()]
    mer = known[known["type"] == "debit"].groupby("merchant")["amount"].sum().sort_values(ascending=False)
    out["top_merchants"] = [(k, float(v)) for k, v in mer.head(10).items()]
    notes: List[str] = []
    for m, vals in months.items():
        inc = income if income > 0 else vals.get("income", 0.0)
        exp = vals.get("expenses", 0.0)
        if inc > 0:
            savings_pct = max(0.0, (inc - exp) / inc * 100.0)
            notes.append(f"{m}: Savings {savings_pct:.1f}% (Income ₹{inc:,.0f}, Expenses ₹{exp:,.0f})")
        else:
            notes.append(f"{m}: Expenses ₹{exp:,.0f} (No income provided/detected)")
    out["notes"] = notes
    return out

# ---------- LLM: Gemini recommendations ----------
def build_llm_context(df: pd.DataFrame, max_rows: int = 800) -> str:
    cols = ["date", "amount", "type", "merchant", "category", "method"]
    slim = df[cols].copy() if not df.empty else pd.DataFrame(columns=cols)
    if len(slim) > max_rows:
        slim = slim.sort_values("date").tail(max_rows)
    return slim.to_json(orient="records", date_format="iso")

@st.cache_data(show_spinner=False)
def generate_recommendations_gemini(df: pd.DataFrame, income: int = 0, max_rows: int = 800) -> str:
    if not GENAI_READY:
        return "⚠️ LLM not configured. Set GEMINI_API_KEY to enable AI insights."
    try:
        context_json = build_llm_context(df, max_rows=max_rows)
        system_msg = "You are a practical, precise personal finance advisor. Use the provided JSON transactions to compute figures."
        user_instructions = {
            "goals": [
                "1) Provide a monthly budget plan (monthly target per category and total monthly budget).",
                "2) Suggest concrete ways to reduce unnecessary or recurring spending with estimated ₹ savings.",
                "3) Provide 3 personalized financial recommendations (e.g., emergency fund, SIPs, debt paydown), with estimated impact in ₹ or % where possible."
            ],
            "notes": "Use transaction data strictly from DATA(JSON). If income is provided, use it for savings calculations; otherwise infer income from credits.",
            "output_format": {
                "sections": ["Monthly Budget Plan", "Reduce Spending Suggestions", "Personalized Advice"],
                "format": "Markdown with bullet lists and numeric estimates. Keep it concise."
            },
            "user_income": income
        }
        prompt = (
            f"SYSTEM:\n{system_msg}\n\n"
            f"DATA(JSON):\n{context_json}\n\n"
            f"INSTRUCTIONS:\n{json.dumps(user_instructions, ensure_ascii=False)}\n\n"
            "Produce the response in Markdown with the three sections exactly as in output_format."
        )
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt)
        return (getattr(resp, 'text', None) or resp or "").strip() if resp else "⚠️ Empty response from LLM."
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower():
            return "⚠️ LLM Quota Exceeded. You've made too many requests. Please try again later."
        return f"⚠️ LLM error: {err_str}"

# ---------- UI ----------
uploaded_files = st.file_uploader("📂 Upload one or more PDF statements", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    files_payload = [(f.name, f.read()) for f in uploaded_files]
    st.success(f"✅ {len(files_payload)} file(s) uploaded.")

    if enable_ocr and not OCR_AVAILABLE:
        st.warning("OCR requested but OCR libraries (pytesseract/pdf2image) are not available. Parsing will proceed without OCR.")

    with st.spinner("📄 Extracting & parsing transactions..."):
        df = extract_and_parse(files_payload, use_ocr=(enable_ocr and OCR_AVAILABLE))

    if df.empty:
        st.error("No transactions parsed. Try enabling OCR or verify the PDF format.")
        # Debug preview (first file)
        try:
            raw_text_dbg = _extract_texts_from_pdf(files_payload[0][1], use_ocr=False)
            st.text_area("🔎 Raw PDF Text Preview (first 2000 chars)", raw_text_dbg[:2000], height=300)
        except Exception:
            pass
    else:
        st.subheader("📋 Parsed Transactions (preview)")
        st.dataframe(df.head(500), use_container_width=True)

        # KPIs
        df_known = df[df["type"].isin(["debit", "credit"])].copy()
        total_exp = float(df_known.loc[df_known["type"] == "debit", "amount"].sum())
        total_inc = float(df_known.loc[df_known["type"] == "credit", "amount"].sum())
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Expenses (₹)", f"{total_exp:,.0f}")
        kpi2.metric("Total Income (₹)", f"{total_inc:,.0f}")
        net = total_inc - total_exp
        kpi3.metric("Net (₹)", f"{net:,.0f}", delta=None)

        # Charts
        tab1, tab2, tab3 = st.tabs(["📆 Monthly", "🗂️ Categories", "🏷️ Merchants"])
        with tab1:
            if "month" in df_known.columns and df_known["month"].nunique() > 0:
                msum = df_known.groupby(["month", "type"]).amount.sum().unstack(fill_value=0)
                st.bar_chart(msum)
        with tab2:
            cat_sum = df_known[df_known["type"] == "debit"].groupby("category").amount.sum().sort_values(ascending=False)
            st.bar_chart(cat_sum)
        with tab3:
            mer_sum = df_known[df_known["type"] == "debit"].groupby("merchant").amount.sum().sort_values(ascending=False).head(20)
            st.bar_chart(mer_sum)

        # Downloads
        st.download_button(
            label="⬇️ Download Parsed CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="transactions_parsed.csv",
            mime="text/csv",
        )

        # Offline insights
        if show_offline_insights:
            st.subheader("📊 Offline Insights (no LLM)")
            off = compute_offline_insights(df, income=user_monthly_income)
            st.json(off)

        # LLM recommendations
        st.subheader("🧠 LLM Recommendations")
        if GENAI_READY:
            if st.button("Generate Recommendations (Gemini)"):
                with st.spinner("Generating recommendations via Gemini..."):
                    md = generate_recommendations_gemini(df, income=user_monthly_income, max_rows=int(chunk_limit))
                st.markdown(md or "⚠️ No insights returned.")
                st.download_button(
                    label="⬇️ Download Recommendations (Markdown)",
                    data=(md or "").encode("utf-8"),
                    file_name="financial_recommendations.md",
                    mime="text/markdown",
                )
                st.markdown('<div class="success-banner">🎉 Recommendations ready — use them to plan your finances!</div>', unsafe_allow_html=True)
                try:
                    st.balloons()
                except Exception:
                    pass
        else:
            st.info("Provide GEMINI_API_KEY in environment to enable AI-powered recommendations.")
else:
    st.info("Upload at least one PDF to begin.")
