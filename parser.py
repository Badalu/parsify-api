import os
import re
import json
import time
import hashlib
import pdfplumber
from pypdf import PdfReader
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

import importlib.util as _ilu

# Prefer the new google-genai SDK; fall back to the old google-generativeai
if _ilu.find_spec("google.genai") is not None:
    from google import genai
    from google.genai import types as genai_types
    GENAI_NEW_SDK = True
else:
    import google.generativeai as genai  # type: ignore[no-redef]
    genai_types = None
    GENAI_NEW_SDK = False

# Initialize Gemini SDK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Increased parallelization — Gemini 2.5 Flash is extremely fast. 
# We split into smaller chunks (approx 2-3 pages) to force parallel execution.
GEMINI_CHUNK_SIZE = 150_000

# Max parallel workers for concurrent Gemini chunk calls
GEMINI_MAX_WORKERS = 2

# Define schemas for structured Gemini output
class TransactionItem(BaseModel):
    date: str = Field(description="The posting date of the transaction (exactly as listed in the statement)")
    value_date: str = Field(description="The value date or clearing date of the transaction if explicitly listed in the statement, else same as date")
    description: str = Field(description="The description or particulars of the transaction")
    debit: str = Field(description="The debit amount (withdrawal), normalized without currency symbols, empty if none")
    credit: str = Field(description="The credit amount (deposit), normalized without currency symbols, empty if none")
    balance: str = Field(description="The running balance after transaction, empty if none")
    category: str = Field(description="Assigned category: Salary, Food, Fuel, Rent, Utilities, Shopping, Groceries, Transfer, Subscription, GST, Tax, Investment, EMI, Refund, Other")
    gst: str = Field(description="If GST splits are found, e.g. 'CGST+SGST', else empty")

class TransactionsList(BaseModel):
    transactions: List[TransactionItem]

class CategorizationItem(BaseModel):
    index: int = Field(description="The index of the transaction in the input list")
    category: str = Field(description="Assigned category: Salary, Food, Fuel, Rent, Utilities, Shopping, Groceries, Transfer, Subscription, GST, Tax, Investment, EMI, Refund, Other")
    gst: str = Field(description="If GST splits are found, e.g. 'CGST+SGST', else empty")

class CategorizationList(BaseModel):
    items: List[CategorizationItem]


def check_pdf_basic(pdf_path: str, password: str = None) -> Tuple[int, bool]:
    """
    Swiftly checks page count and whether the document is text-based by reading just the first two pages.
    Returns (total_pages, is_text_based).
    """
    try:
        reader = PdfReader(pdf_path, password=password)
        total_pages = len(reader.pages)
        if total_pages == 0:
            return 0, False
        
        # Check up to 2 pages to verify it's not just a scanned image
        for i in range(min(2, total_pages)):
            text = reader.pages[i].extract_text()
            if text and len(text.strip()) > 20:
                return total_pages, True
        return total_pages, False
    except Exception as e:
        raise e

def extract_full_text(pdf_path: str, password: str = None) -> str:
    """
    Extracts full text from all pages sequentially using pypdf for maximum speed,
    with a robust fallback to pdfplumber layout-aware text extraction.
    """
    try:
        reader = PdfReader(pdf_path, password=password)
        extracted_pages = []
        for page in reader.pages:
            extracted_pages.append(page.extract_text() or "")
        return "\n\n--- Page Break ---\n\n".join(extracted_pages)
    except Exception as e:
        print(f"Error during pypdf text extraction: {e} — falling back to pdfplumber")
        extracted_pages = []
        try:
            with pdfplumber.open(pdf_path, password=password) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text(layout=True)
                        if text:
                            extracted_pages.append(text)
                    except Exception:
                        pass
                    finally:
                        page.flush_cache()
            return "\n\n--- Page Break ---\n\n".join(extracted_pages)
        except Exception as e2:
            print(f"Fallback pdfplumber text extraction failed: {e2}")
            return ""


# ── Native Rule-Based Parser Helpers ──────────────────────────────────────────

DATE_PATTERNS = [
    re.compile(r'^\s*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\s*$'),          # 12/05/2023, 12-05-23
    re.compile(r'^\s*\d{1,2}[-/\.][A-Za-z]{3,4}[-/\.]\d{2,4}\s*$'),    # 12-May-2023, 12-May-23
    re.compile(r'^\s*\d{1,2}\s+[A-Za-z]{3,4}\s+\d{2,4}\s*$'),           # 12 May 2023, 12 May 23
    re.compile(r'^\s*[A-Za-z]{3,4}\s+\d{1,2},?\s+\d{2,4}\s*$'),        # May 12, 2023, May 12 23
    re.compile(r'^\s*\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\s*$'),            # 2023-05-12
]

def is_valid_date(val: str) -> bool:
    if not val:
        return False
    val_str = str(val).strip()
    for pattern in DATE_PATTERNS:
        if pattern.match(val_str):
            return True
    return False


def categorize_locally(description: str) -> str:
    desc_lower = str(description).lower()

    categories = {
        "Salary": ["salary", "payroll", "wage", "direct dep", "neft salary"],
        "Food": ["zomato", "swiggy", "restaurant", "cafe", "food", "dining", "domino", "starbuck"],
        "Fuel": ["petrol", "diesel", "hpcl", "bpcl", "iocl", "fuel", "shell"],
        "Rent": ["rent", "landlord", "brokerage", "house rent"],
        "Utilities": ["electricity", "bescom", "tneb", "uppcl", "water", "broadband", "jio", "airtel", "vi ", "recharge", "gas", "indane", "act fibernet"],
        "Shopping": ["amazon", "flipkart", "myntra", "meesho", "clothing", "appliances", "retail"],
        "Groceries": ["blinkit", "zepto", "instamart", "bigbasket", "supermarket", "grocery", "groceries", "milk", "dairy"],
        "Transfer": ["upi", "transfer", "neft", "rtgs", "imps", "ft", "sent to", "received from", "gpay", "phonepe"],
        "Subscription": ["netflix", "prime video", "spotify", "youtube premium", "disney", "hotstar", "microsoft", "google storage", "cloud"],
        "GST": ["gst", "cgst", "sgst", "igst", "tax split"],
        "Tax": ["income tax", "tds", "itr", "advance tax", "professional tax"],
        "Investment": ["zerodha", "groww", "mutual fund", "sip", "demat", "shares", "stocks", "etf"],
        "EMI": ["emi", "loan", "mortgage", "hdfc bank loan", "sbi loan", "car emi", "home loan"],
        "Refund": ["refund", "cashback", "returned", "reversal"]
    }

    for cat, keywords in categories.items():
        if any(kw in desc_lower for kw in keywords):
            return cat

    return "Other"


def extract_gst_locally(description: str) -> str:
    desc_lower = str(description).lower()
    if "cgst" in desc_lower and "sgst" in desc_lower:
        return "CGST+SGST"
    elif "igst" in desc_lower:
        return "IGST"
    elif "gst" in desc_lower:
        return "GST"
    return ""


def find_header_mapping(row: List[str]) -> Dict[str, int]:
    mapping = {}
    row_lower = [str(cell).lower().strip() if cell is not None else "" for cell in row]

    date_kws = ["date", "dt", "txn d", "trans d"]
    desc_kws = ["particulars", "description", "narration", "remarks", "transaction details", "details", "narrative"]
    debit_kws = ["debit", "withdrawal", "payment", "dr", "withdraw"]
    credit_kws = ["credit", "deposit", "receipt", "cr"]
    balance_kws = ["balance", "bal", "running"]

    for idx, cell in enumerate(row_lower):
        if not cell:
            continue

        if any(kw in cell for kw in date_kws):
            if "value" not in cell and "post" not in cell and "val" not in cell:
                mapping["date"] = idx
            elif "value" in cell or "val" in cell:
                mapping["value_date"] = idx

        if any(kw in cell for kw in desc_kws):
            mapping["description"] = idx

        if any(kw in cell for kw in debit_kws):
            mapping["debit"] = idx

        if any(kw in cell for kw in credit_kws):
            mapping["credit"] = idx

        if any(kw in cell for kw in balance_kws):
            mapping["balance"] = idx

    if "date" in mapping and "description" in mapping:
        return mapping
    return {}


def parse_line_by_line(text: str) -> List[Dict[str, Any]]:
    """
    Fallback parser that extracts transactions from raw page text line-by-line.
    Highly robust against borderless table layouts.
    """
    transactions = []
    lines = text.split("\n")

    header_bounds = {}

    date_kws = ["date", "dt", "txn d", "trans d"]
    desc_kws = ["particulars", "description", "narration", "remarks", "transaction details", "details", "narrative"]
    debit_kws = ["debit", "withdrawal", "payment", "dr", "withdraw", "withdrawal(dr)"]
    credit_kws = ["credit", "deposit", "receipt", "cr", "deposit(cr)"]
    balance_kws = ["balance", "bal", "running"]

    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in date_kws) and any(kw in line_lower for kw in desc_kws):
            bounds = {}

            for kw in date_kws:
                idx = line_lower.find(kw)
                if idx != -1:
                    if "value" in line_lower[max(0, idx-10):idx+20] or "val" in line_lower[max(0, idx-10):idx+20]:
                        bounds["value_date"] = idx
                    else:
                        bounds["date"] = idx

            if "date" not in bounds:
                for kw in date_kws:
                    idx = line_lower.find(kw)
                    if idx != -1:
                        bounds["date"] = idx
                        break

            for kw in desc_kws:
                idx = line_lower.find(kw)
                if idx != -1:
                    bounds["description"] = idx
                    break

            for kw in debit_kws:
                idx = line_lower.find(kw)
                if idx != -1:
                    bounds["debit"] = idx
                    break

            for kw in credit_kws:
                idx = line_lower.find(kw)
                if idx != -1:
                    bounds["credit"] = idx
                    break

            for kw in balance_kws:
                idx = line_lower.find(kw)
                if idx != -1:
                    bounds["balance"] = idx
                    break

            if "date" in bounds and "description" in bounds:
                sorted_cols = sorted(bounds.items(), key=lambda x: x[1])
                header_bounds = {}
                for i in range(len(sorted_cols)):
                    col_name, start = sorted_cols[i]
                    col_start = start
                    if i > 0:
                        prev_name, prev_start = sorted_cols[i-1]
                        col_start = (prev_start + col_start) // 2 + 2
                    else:
                        col_start = 0

                    col_end = sorted_cols[i+1][1] if i < len(sorted_cols) - 1 else 300
                    header_bounds[col_name] = (col_start, col_end)
                break

    for line in lines:
        if not line.strip():
            continue

        line_lower = line.lower()
        if any(kw in line_lower for kw in date_kws) and any(kw in line_lower for kw in desc_kws):
            continue

        if header_bounds:
            date_idx = header_bounds.get("date")
            desc_idx = header_bounds.get("description")
            debit_idx = header_bounds.get("debit")
            credit_idx = header_bounds.get("credit")
            balance_idx = header_bounds.get("balance")
            val_date_idx = header_bounds.get("value_date")

            def get_slice(b):
                if not b:
                    return ""
                s, e = b
                return line[s:min(e, len(line))].strip() if s < len(line) else ""

            date_val = get_slice(date_idx)
            desc_val = get_slice(desc_idx)

            if is_valid_date(date_val):
                debit_val = get_slice(debit_idx)
                credit_val = get_slice(credit_idx)
                balance_val = get_slice(balance_idx)
                val_date_val = get_slice(val_date_idx)

                if not desc_val:
                    continue

                transactions.append({
                    "date": date_val,
                    "value_date": val_date_val if val_date_val else date_val,
                    "description": desc_val,
                    "debit": debit_val,
                    "credit": credit_val,
                    "balance": balance_val,
                    "category": categorize_locally(desc_val),
                    "gst": extract_gst_locally(desc_val)
                })
                continue

            elif desc_val and transactions:
                skip_words = ["balance", "carried", "brought", "total", "page", "statement", "generated on", "opening", "closing"]
                if not any(sw in desc_val.lower() for sw in skip_words):
                    transactions[-1]["description"] += " " + desc_val

                    debit_val = get_slice(debit_idx)
                    credit_val = get_slice(credit_idx)
                    balance_val = get_slice(balance_idx)

                    if debit_val and not transactions[-1]["debit"]:
                        transactions[-1]["debit"] = debit_val
                    if credit_val and not transactions[-1]["credit"]:
                        transactions[-1]["credit"] = credit_val
                    if balance_val and not transactions[-1]["balance"]:
                        transactions[-1]["balance"] = balance_val
                continue

        parts = re.split(r'\s{2,}', line.strip())
        if not parts:
            continue

        date_val = ""
        date_part_idx = -1
        for idx, p in enumerate(parts):
            if is_valid_date(p):
                date_val = p
                date_part_idx = idx
                break

        if date_val and date_part_idx != -1:
            remaining = parts[date_part_idx+1:]
            if not remaining:
                continue

            value_date_val = date_val
            if is_valid_date(remaining[0]):
                value_date_val = remaining.pop(0)

            if not remaining:
                continue

            amounts = []
            desc_parts = []

            def is_amount(s):
                s_clean = s.replace(",", "").replace("+", "").replace("-", "").strip()
                # Remove Dr/Cr suffixes
                s_clean = re.sub(r'(?i)(dr|cr)$', '', s_clean).strip()
                if not s_clean:
                    return False
                try:
                    float(s_clean)
                    return True
                except ValueError:
                    return False

            for r in remaining:
                if is_amount(r):
                    amounts.append(r)
                else:
                    desc_parts.append(r)

            desc_val = " ".join(desc_parts).strip()
            if not desc_val:
                continue

            debit_val = ""
            credit_val = ""
            balance_val = ""

            if len(amounts) == 1:
                desc_lower = desc_val.lower()
                is_deb = any(kw in desc_lower for kw in ["payment", "upi to", "transfer to", "dr", "debit", "withdrawal"])
                if is_deb:
                    debit_val = amounts[0]
                else:
                    credit_val = amounts[0]
            elif len(amounts) == 2:
                amt = amounts[0]
                bal = amounts[1]

                desc_lower = desc_val.lower()
                is_deb = any(kw in desc_lower for kw in ["payment", "upi to", "transfer to", "dr", "debit", "withdrawal"]) or amt.startswith("-")

                if is_deb:
                    debit_val = amt
                else:
                    credit_val = amt
                balance_val = bal
            elif len(amounts) == 3:
                debit_val = amounts[0]
                credit_val = amounts[1]
                balance_val = amounts[2]

            transactions.append({
                "date": date_val,
                "value_date": value_date_val,
                "description": desc_val,
                "debit": debit_val,
                "credit": credit_val,
                "balance": balance_val,
                "category": categorize_locally(desc_val),
                "gst": extract_gst_locally(desc_val)
            })

        elif len(parts) == 1 and transactions:
            val = parts[0]
            skip_words = ["balance", "carried", "brought", "total", "page", "statement", "generated on", "opening", "closing"]
            if not any(sw in val.lower() for sw in skip_words) and len(val) > 3:
                transactions[-1]["description"] += " " + val

    return transactions


def parse_pdf_natively(pdf_path: str, password: str = None) -> List[Dict[str, Any]]:
    """
    Extract transactions from structured PDF statements natively using pdfplumber without Gemini.
    """
    start_time = time.time()
    
    # Check page count first. Large PDFs are bypassed on Vercel to avoid 10s gateway timeouts.
    # Gemini fallback processes pages in parallel and runs within 4-5s total.
    is_vercel = os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV") is not None
    max_pages = 12 if is_vercel else 100
    
    try:
        reader = PdfReader(pdf_path, password=password)
        total_pages = len(reader.pages)
        if total_pages > max_pages:
            print(f"[Native] PDF has {total_pages} pages (limit: {max_pages} pages). Bypassing native parsing to avoid timeouts.")
            return []
    except Exception as e:
        print(f"[Native] Failed to check page count via pypdf: {e}")
        # Proceed to try opening with pdfplumber

    transactions: List[Dict[str, Any]] = []
    header_mapping: Dict[str, int] = {}

    try:
        with pdfplumber.open(pdf_path, password=password) as pdf:
            table_found = False
            for idx, page in enumerate(pdf.pages):
                # Timeout check to prevent gateway timeouts on Vercel
                if time.time() - start_time > 6.0:
                    print(f"[Native] Timeout of 6.0s exceeded at page {idx+1}/{len(pdf.pages)}. Aborting native table parser.")
                    return []

                if idx >= 2 and not table_found:
                    print("[Native] No tables found in first 2 pages. Aborting table extraction for speed.")
                    break
                tables = page.extract_tables()
                if not tables:
                    try:
                        tables = [t.extract() for t in page.find_tables()]
                    except Exception:
                        tables = []
                if tables:
                    table_found = True

                for table in tables:
                    if not table:
                        continue

                    for row in table:
                        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]

                        if not any(clean_row):
                            continue

                        mapping = find_header_mapping(clean_row)
                        if mapping:
                            header_mapping = mapping
                            continue

                        if not header_mapping:
                            continue

                        date_idx = header_mapping.get("date")
                        desc_idx = header_mapping.get("description")
                        debit_idx = header_mapping.get("debit")
                        credit_idx = header_mapping.get("credit")
                        balance_idx = header_mapping.get("balance")
                        val_date_idx = header_mapping.get("value_date")

                        date_val = clean_row[date_idx] if date_idx is not None and date_idx < len(clean_row) else ""
                        desc_val = clean_row[desc_idx] if desc_idx is not None and desc_idx < len(clean_row) else ""

                        if not desc_val and not date_val:
                            continue

                        if is_valid_date(date_val):
                            debit_val = clean_row[debit_idx] if debit_idx is not None and debit_idx < len(clean_row) else ""
                            credit_val = clean_row[credit_idx] if credit_idx is not None and credit_idx < len(clean_row) else ""
                            balance_val = clean_row[balance_idx] if balance_idx is not None and balance_idx < len(clean_row) else ""
                            val_date_val = clean_row[val_date_idx] if val_date_idx is not None and val_date_idx < len(clean_row) else ""

                            # Repair column shift caused by missing Value Date column in borderless tables
                            if val_date_idx is not None and val_date_val and len(val_date_val) > 11 and not is_valid_date(val_date_val) and re.search(r'[A-Za-z]{2,}', val_date_val):
                                balance_val = clean_row[credit_idx] if credit_idx is not None and credit_idx < len(clean_row) else ""
                                credit_val = clean_row[debit_idx] if debit_idx is not None and debit_idx < len(clean_row) else ""
                                debit_val = clean_row[desc_idx] if desc_idx is not None and desc_idx < len(clean_row) else ""
                                desc_val = val_date_val
                                val_date_val = date_val

                            # If debit is still alphabetic (not an amount), description spilled over
                            if debit_val and re.search(r'[A-Za-z]{3,}', debit_val) and "dr" not in debit_val.lower():
                                desc_val += " " + debit_val
                                debit_val = credit_val
                                credit_val = balance_val
                                balance_val = clean_row[balance_idx+1] if balance_idx is not None and balance_idx+1 < len(clean_row) else ""

                            txn = {
                                "date": date_val,
                                "value_date": val_date_val if val_date_val else date_val,
                                "description": desc_val,
                                "debit": debit_val,
                                "credit": credit_val,
                                "balance": balance_val,
                                "category": categorize_locally(desc_val),
                                "gst": extract_gst_locally(desc_val)
                            }
                            transactions.append(txn)

                        elif desc_val and not date_val and transactions:
                            desc_lower = desc_val.lower()
                            skip_words = ["balance", "carried", "brought", "total", "page", "statement"]
                            if not any(sw in desc_lower for sw in skip_words):
                                transactions[-1]["description"] += " " + desc_val

                                if debit_idx is not None and debit_idx < len(clean_row) and clean_row[debit_idx] and not transactions[-1]["debit"]:
                                    transactions[-1]["debit"] = clean_row[debit_idx]
                                if credit_idx is not None and credit_idx < len(clean_row) and clean_row[credit_idx] and not transactions[-1]["credit"]:
                                    transactions[-1]["credit"] = clean_row[credit_idx]
                                if balance_idx is not None and balance_idx < len(clean_row) and clean_row[balance_idx] and not transactions[-1]["balance"]:
                                    transactions[-1]["balance"] = clean_row[balance_idx]

                page.flush_cache()

        if not transactions:
            print("[Native] No transactions found using table extraction. Trying line-by-line regex parser fallback...")
            full_text_list = []
            with pdfplumber.open(pdf_path, password=password) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        full_text_list.append(page_text)
                    page.flush_cache()
            if full_text_list:
                raw_text = "\n".join(full_text_list)
                transactions = parse_line_by_line(raw_text)

    except Exception as e:
        print(f"Error during native PDF parsing: {e}")

    return transactions


def _build_system_prompt(categorize: bool, gst: bool) -> str:
    """
    Build a comprehensive system prompt for accurate bank statement parsing.
    Works across all major Indian banks (SBI, HDFC, ICICI, Axis, Kotak, PNB, BOB,
    Canara, IDFC, Yes Bank, IndusInd) and international formats.
    """
    prompt = (
        "You are a world-class bank statement parser with deep expertise in Indian "
        "and international bank statement formats. Your sole job is to extract EVERY "
        "transaction from the provided bank statement text with 100% accuracy.\n\n"
        "=== CRITICAL RULES ===\n"
        "- NEVER skip a transaction. Even partial or unclear rows must be included.\n"
        "- NEVER hallucinate data. Only extract information explicitly present.\n"
        "- NEVER merge two separate transactions into one.\n"
        "- Process transactions in chronological order, exactly as they appear.\n"
        "- Tab-separated rows indicate structured table data — treat each tab as a column separator.\n\n"
        "=== FIELD EXTRACTION RULES ===\n"
        "1. DATE: Extract EXACTLY as shown in the statement (e.g., '01/05/2024', '01-May-24', '1 May 2024').\n"
        "   - Do NOT reformat or convert the date.\n"
        "   - Column headers may say: Date, Txn Date, Transaction Date, Posting Date, Value Date, Dt.\n\n"
        "2. VALUE_DATE: The clearing/value date if listed in a separate column (e.g., 'Value Dt', 'Val Date').\n"
        "   - If no separate value date column exists, copy the transaction date.\n\n"
        "3. DESCRIPTION: The full transaction narrative/particulars.\n"
        "   - Column headers may say: Particulars, Narration, Description, Remarks, Transaction Details, Chq/Ref No.\n"
        "   - If the description spans multiple lines, concatenate them with a space.\n"
        "   - Include UPI IDs, reference numbers, merchant names, and all visible details.\n\n"
        "4. DEBIT: Any withdrawal, payment, or outflow (money leaving the account).\n"
        "   - Column headers: Debit, Withdrawal, Dr, Payment, Dr Amount, Withdrawal(Dr).\n"
        "   - Normalize: Remove ₹, Rs., INR, USD, commas. Remove 'Dr'/'Cr' suffix from amounts.\n"
        "   - If empty/nil/-, set to empty string ''.\n\n"
        "5. CREDIT: Any deposit, receipt, or inflow (money entering the account).\n"
        "   - Column headers: Credit, Deposit, Cr, Receipt, Cr Amount, Deposit(Cr).\n"
        "   - Same normalization as debit.\n"
        "   - If empty/nil/-, set to empty string ''.\n\n"
        "6. BALANCE: Running balance after the transaction.\n"
        "   - Column headers: Balance, Bal, Running Balance, Closing Balance.\n"
        "   - If not present in the statement, set to empty string ''.\n\n"
        "=== BANK-SPECIFIC NOTES ===\n"
        "- SBI: Columns are typically Txn Date, Value Date, Description, Ref No/Cheque No, Debit, Credit, Balance.\n"
        "- HDFC: Columns are Date, Narration, Value Dt, Chq/Ref No, Withdrawal Amt, Deposit Amt, Closing Balance.\n"
        "- ICICI: Columns are Transaction Date, Value Date, Particulars, Cheque No, Deposits, Withdrawals, Balance.\n"
        "- Axis: Columns are Tran Date, Chq No, Particulars, Debit, Credit, Balance.\n"
        "- Kotak: Columns are Date, Description, Chq/Ref No, Debit, Credit, Balance.\n"
        "- IDFC/IndusInd/Yes Bank: May use compact formats with Dr/Cr column style.\n\n"
        "=== WHAT TO SKIP ===\n"
        "- Header rows (containing column names)\n"
        "- Footer rows (totals, page numbers, bank addresses)\n"
        "- Opening Balance / Closing Balance summary lines\n"
        "- Blank rows\n\n"
    )

    if categorize:
        prompt += (
            "=== CATEGORY ASSIGNMENT ===\n"
            "Assign exactly ONE category per transaction from this list ONLY:\n"
            "Salary, Food, Fuel, Rent, Utilities, Shopping, Groceries, Transfer, "
            "Subscription, GST, Tax, Investment, EMI, Refund, Other\n"
            "Rules:\n"
            "- Salary: NEFT salary credits, payroll, ECS salary\n"
            "- Food: Zomato, Swiggy, Dunzo, restaurants, cafes, dining\n"
            "- Fuel: Petrol, diesel, HPCL, BPCL, IOCL, Shell, HP, BP\n"
            "- Rent: House rent, flat rent, PG rent, accommodation\n"
            "- Utilities: Electricity (BESCOM/TNEB/UPPCL), water, gas, Jio, Airtel, Vi, broadband\n"
            "- Shopping: Amazon, Flipkart, Myntra, Meesho, retail stores, clothing\n"
            "- Groceries: Blinkit, Zepto, BigBasket, Dunzo grocery, supermarket, dairy\n"
            "- Transfer: UPI, NEFT, RTGS, IMPS, inter-account transfer, GPay, PhonePe, Paytm\n"
            "- Subscription: Netflix, Spotify, Amazon Prime, YouTube Premium, Disney+, Microsoft\n"
            "- GST: GST payment, CGST, SGST, IGST entries\n"
            "- Tax: Income tax, TDS, advance tax, professional tax\n"
            "- Investment: Zerodha, Groww, Kuvera, SIP, mutual fund, demat, stocks\n"
            "- EMI: Loan EMI, car EMI, home loan EMI, personal loan repayment\n"
            "- Refund: Refund, cashback, reversal, returned\n"
            "- Other: All unclassified transactions\n\n"
        )
    else:
        prompt += "=== CATEGORY ===\nSet category to 'Other' for every row.\n\n"

    if gst:
        prompt += (
            "=== GST FIELD ===\n"
            "If the transaction description explicitly mentions GST components, set:\n"
            "- 'CGST+SGST' if both Central and State GST are mentioned\n"
            "- 'IGST' if Integrated GST is mentioned\n"
            "- 'GST' if only generic GST is mentioned\n"
            "- '' (empty) for all other transactions\n"
        )
    else:
        prompt += "=== GST FIELD ===\nSet gst to '' (empty string) for all rows.\n"

    return prompt


def _call_gemini_chunk(
    model_or_client,
    system_prompt: str,
    text_chunk: str,
    chunk_num: int = 1,
    chunk_context: str = ""
) -> List[Dict[str, Any]]:
    """
    Call Gemini with a single chunk of text, with retry + exponential backoff.
    Supports both new google.genai and legacy google.generativeai SDK.
    """
    user_message = (
        f"Extract ALL transactions from this bank statement text. "
        f"Return every transaction row, do not skip any.\n"
        f"{chunk_context}\n"
        f"BANK STATEMENT TEXT:\n{'='*60}\n{text_chunk}\n{'='*60}"
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            if GENAI_NEW_SDK:
                client = model_or_client
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_message,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=TransactionsList,
                        temperature=0,  # 0 = deterministic, max accuracy
                    )
                )
                raw = response.text
            else:
                model = model_or_client
                response = model.generate_content(
                    contents=[{"role": "user", "parts": [user_message]}],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=TransactionsList,
                        temperature=0,
                    )
                )
            raw = response.text
            # Clean up potential markdown formatting
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            data = json.loads(raw)
            txns = data.get("transactions", [])
            print(f"  Chunk {chunk_num}: Extracted {len(txns)} transactions")
            return txns
        except Exception as e:
            print(f"  Chunk {chunk_num} attempt {attempt+1} failed: {e}")
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg or "limit" in err_msg:
                print(f"Rate limit / Quota hit on chunk {chunk_num}. Waiting longer before retry...")
                time.sleep(8) # Force a longer sleep for rate limits
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s, 8s
            else:
                raise
    return []


def _deduplicate_transactions(txns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact duplicate transaction rows that may appear when chunks overlap.
    Uses a fingerprint of (date, description, debit, credit, balance).
    """
    seen = set()
    result = []
    for txn in txns:
        # Build a stable fingerprint
        key = hashlib.md5(
            f"{txn.get('date','')}__{txn.get('description','')}__{txn.get('debit','')}__{txn.get('credit','')}__{txn.get('balance','')}".encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(txn)
    removed = len(txns) - len(result)
    if removed:
        print(f"  Deduplication: removed {removed} exact duplicate transactions")
    return result


def parse_with_gemini(text: str, categorize: bool = True, gst: bool = True) -> List[Dict[str, Any]]:
    """
    Send statement text to Google Gemini and get parsed transactions in structured JSON.
    Handles large statements by splitting into chunks and merging results IN PARALLEL.
    Uses a comprehensive, bank-specific prompt for 100% accuracy.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not configured.")

    system_prompt = _build_system_prompt(categorize, gst)

    if GENAI_NEW_SDK:
        model_or_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Using new google.genai SDK")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model_or_client = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )
        print("Using legacy google.generativeai SDK (consider upgrading)")

    all_transactions = []

    if len(text) <= GEMINI_CHUNK_SIZE:
        # Single call for normal-sized statements
        all_transactions = _call_gemini_chunk(model_or_client, system_prompt, text, chunk_num=1)
    else:
        # Multi-chunk approach: split by page breaks first, then by size
        print(f"Large statement ({len(text)} chars), splitting into parallel chunks...")
        pages = text.split("\n\n--- Page Break ---\n\n")

        # Build chunk list
        chunks: List[Tuple[int, str]] = []  # (chunk_num, text)
        current_chunk = ""
        chunk_num = 1

        for page in pages:
            if len(current_chunk) + len(page) > GEMINI_CHUNK_SIZE and current_chunk:
                chunks.append((chunk_num, current_chunk))
                chunk_num += 1
                current_chunk = page
            else:
                current_chunk += ("\n\n--- Page Break ---\n\n" if current_chunk else "") + page

        if current_chunk:
            chunks.append((chunk_num, current_chunk))

        total_chunks = len(chunks)
        print(f"  Processing {total_chunks} chunks sequentially to respect rate limits...")

        chunk_results: Dict[int, List[Dict[str, Any]]] = {}

        for c_num, chunk_text in chunks:
            if c_num > 1:
                # 2-second safety delay between chunks to stay under 15 RPM
                time.sleep(2.0)
            
            print(f"  Processing chunk {c_num} of {total_chunks}...")
            txns = _call_gemini_chunk(
                model_or_client,
                system_prompt,
                chunk_text,
                c_num,
                f"(Chunk {c_num} of {total_chunks} — process only the transactions in this section)"
            )
            chunk_results[c_num] = txns

        # Merge in order
        for c_num in sorted(chunk_results.keys()):
            all_transactions.extend(chunk_results[c_num])

    # Normalize output
    result = []
    for txn in all_transactions:
        if isinstance(txn, dict):
            result.append(txn)
        else:
            result.append(txn.model_dump() if hasattr(txn, 'model_dump') else dict(txn))

    # Deduplicate (catches cross-chunk overlapping transactions)
    result = _deduplicate_transactions(result)

    print(f"Gemini total extracted: {len(result)} transactions")
    return result


def categorize_and_tag_with_gemini(
    transactions: List[Dict[str, Any]],
    categorize: bool = True,
    gst: bool = True
) -> List[Dict[str, Any]]:
    """
    Categorize transactions and tag GST using Gemini 2.0 Flash in batches.
    Falls back to local rules defensively if the API call fails.
    """
    if not transactions:
        return []
    if not GEMINI_API_KEY:
        print("[Gemini] API Key not set, using local rule-based categories.")
        return transactions

    system_prompt = (
        "You are a highly accurate financial transactions categorizer. Your task is to assign a category "
        "and detect GST for each transaction description provided.\n\n"
        "Categories to choose from:\n"
        "Salary, Food, Fuel, Rent, Utilities, Shopping, Groceries, Transfer, Subscription, GST, Tax, Investment, EMI, Refund, Other\n\n"
        "GST detection:\n"
        "- 'CGST+SGST' if description mentions CGST and SGST\n"
        "- 'IGST' if IGST is mentioned\n"
        "- 'GST' if generic GST is mentioned\n"
        "- '' (empty string) if no GST is mentioned\n\n"
        "Ensure you return a list of objects containing the index of the transaction, the category, and the gst value."
    )

    # Prepare input list: only send index and description to minimize token usage
    input_items = [{"index": idx, "description": txn.get("description", "")} for idx, txn in enumerate(transactions)]

    # Split into chunks of 150 to stay well within limits
    chunk_size = 150
    chunks = [input_items[i:i + chunk_size] for i in range(0, len(input_items), chunk_size)]
    
    if GENAI_NEW_SDK:
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    categorized_items = {}

    for c_idx, chunk in enumerate(chunks):
        user_message = f"Process this chunk of transactions:\n{json.dumps(chunk)}"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if GENAI_NEW_SDK:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=user_message,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=CategorizationList,
                            temperature=0,
                        )
                    )
                    raw = response.text
                else:
                    response = model.generate_content(
                        contents=[{"role": "user", "parts": [user_message]}],
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=CategorizationList,
                            temperature=0,
                        )
                    )
                    raw = response.text
                
                raw = raw.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
                
                data = json.loads(raw)
                for item in data.get("items", []):
                    idx = item.get("index")
                    if idx is not None:
                        categorized_items[int(idx)] = {
                            "category": item.get("category", "Other") if categorize else "Other",
                            "gst": item.get("gst", "") if gst else ""
                        }
                break  # Success
            except Exception as e:
                print(f"[Gemini Categorize] Chunk {c_idx+1} attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[Gemini Categorize] Falling back to local rules for chunk {c_idx+1}")
                    for item in chunk:
                        idx = item["index"]
                        categorized_items[idx] = {
                            "category": transactions[idx].get("category", "Other"),
                            "gst": transactions[idx].get("gst", "")
                        }

    # Merge back into transactions
    for idx, txn in enumerate(transactions):
        cat_info = categorized_items.get(idx)
        if cat_info:
            txn["category"] = cat_info["category"]
            txn["gst"] = cat_info["gst"]

    return transactions



def clean_and_format_transactions(txns: List[Dict[str, Any]], date_format: str = "DD/MM/YYYY") -> List[Dict[str, Any]]:
    """
    Use pandas to standardize data formats, clean amounts, and sort.
    """
    if not txns:
        return []

    df = pd.DataFrame(txns)

    # Fill NaN values with empty strings
    df = df.fillna("")

    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()

    def clean_amount(val):
        if not val or val in ("—", "None", "null", "nil", "N/A", "n/a"):
            return ""
        # Remove currency symbols: ₹, Rs., INR, USD, $, £, €
        cleaned = re.sub(r'(?i)(rs\.?|inr|usd|₹|\$|£|€)', '', str(val))
        # Remove Dr/Cr suffix (case-insensitive)
        cleaned = re.sub(r'(?i)\s*(dr|cr)\s*$', '', cleaned)
        # Keep only digits, decimal point, comma (thousands sep), and leading minus
        cleaned = re.sub(r'[^\d\.,\-]', '', cleaned)
        # Collapse multiple dots/commas
        if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = '.'.join([parts[0].replace(',', ''), parts[-1]])
        return cleaned.strip() if cleaned.strip() not in ('', '-', '.') else ""

    df['debit'] = df['debit'].apply(clean_amount)
    df['credit'] = df['credit'].apply(clean_amount)
    df['balance'] = df['balance'].apply(clean_amount)

    if 'description' in df.columns:
        df['description'] = df['description'].apply(lambda x: x.strip())

    return df.to_dict(orient="records")


def generate_excel_file(txns: List[Dict[str, Any]], file_path: str):
    """
    Create a highly styled, professional Excel worksheet from transactions list using pandas and openpyxl.
    """
    if not txns:
        return

    df = pd.DataFrame(txns)

    cols = ["date", "value_date", "description", "debit", "credit", "balance", "category", "gst"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    rename_dict = {
        "date": "Date",
        "value_date": "Value Date",
        "description": "Description",
        "debit": "Debit",
        "credit": "Credit",
        "balance": "Balance",
        "category": "Category",
        "gst": "GST"
    }
    df = df.rename(columns=rename_dict)

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Transactions", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Transactions"]

        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter

        header_fill = PatternFill(start_color="4F46E5", end_color="4F46E5", fill_type="solid")
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")

        zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
        border_side = Side(border_style="thin", color="E2E8F0")
        thin_border = Border(left=border_side, right=border_side, top=border_side, bottom=border_side)

        cell_font = Font(name="Segoe UI", size=10)

        for col_idx in range(1, len(df.columns) + 1):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center" if col_idx != 2 else "left", vertical="center")
            cell.border = thin_border

        for row_idx in range(2, worksheet.max_row + 1):
            is_zebra = (row_idx % 2 == 0)
            for col_idx in range(1, len(df.columns) + 1):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                cell.font = cell_font
                cell.border = thin_border

                if is_zebra:
                    cell.fill = zebra_fill

                col_name = df.columns[col_idx - 1]
                if col_name in ["Debit", "Credit", "Balance"]:
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                    try:
                        val_str = str(cell.value).replace(",", "")
                        if val_str and val_str not in ("—", "None", ""):
                            cell.value = float(val_str)
                            cell.number_format = '#,##0.00'
                    except ValueError:
                        pass
                elif col_name == "Date":
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

        for col in worksheet.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or "")
                if len(val_str) > max_len:
                    max_len = len(val_str)
            worksheet.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 50)

        worksheet.row_dimensions[1].height = 28
        for r in range(2, worksheet.max_row + 1):
            worksheet.row_dimensions[r].height = 20


def generate_csv_file(txns: List[Dict[str, Any]], file_path: str):
    """
    Generate standard CSV file from transactions.
    """
    if not txns:
        return
    df = pd.DataFrame(txns)
    cols = ["date", "value_date", "description", "debit", "credit", "balance", "category", "gst"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df.to_csv(file_path, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compat
