import os
import shutil
import tempfile
import asyncio
import json
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List

from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from parser import (
    check_pdf_basic,
    decrypt_pdf_if_needed,
    extract_full_text,
    parse_pdf_natively,
    parse_with_gemini,
    parse_line_by_line,
    clean_and_format_transactions,
    generate_excel_file,
    generate_csv_file,
    categorize_and_tag_with_gemini,
    parse_file_directly_with_gemini,
    parse_large_pdf_chunked,
    parse_bank_statement_smart,
    is_text_quality_sufficient,
    is_valid_date,
    detect_bank_name,
    detect_account_details,
    PDF_CHUNK_THRESHOLD,
    reconcile_and_heal_transactions,
)
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Failed to create Supabase client: {e}")

app = FastAPI(title="StatementConvert Python API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://parsify.in",
        "https://www.parsify.in",
        "https://app.parsify.in",
        "https://convertor.vercel.app",
        "https://convertor-pgzqlnnyc-badal-s-projects1.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/test-timeout")
async def test_timeout():
    import asyncio
    await asyncio.sleep(15)
    return {"status": "success", "message": "Waited 15 seconds successfully"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print(traceback.format_exc())
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        }
    )


# ─── Constants ────────────────────────────────────────────────────────────────
ANON_STMT_LIMIT            = 10   # anonymous users: 10 statements/conversions per day
REGISTERED_STMT_LIMIT      = 15   # free registered users: 15 statements/conversions per day
STARTER_STMT_LIMIT         = 40   # starter plan: 40 statements/month
GROWTH_STMT_LIMIT          = 120  # growth plan: 120 statements/month
PRO_STMT_LIMIT             = 400  # pro plan: 400 statements/month
MAX_BATCH_FILES            = 20   # max files per batch request
MAX_FILE_SIZE_MB           = 100  # max single file size in MB


import httpx

# ─── Auth helper ─────────────────────────────────────────────────────────────
async def verify_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization or not supabase:
        return None
    try:
        if not authorization.startswith("Bearer "):
            return None
        token = authorization.split(" ")[1]
        user_res = supabase.auth.get_user(token)
        if user_res and user_res.user:
            return {
                "id": user_res.user.id,
                "email": user_res.user.email,
                "role": user_res.user.role,
                "token": token
            }
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
    return None


# ─── Quota helper — fetches profile + usage via HTTP using user token ────────
#
# Quota model:
#   Free (registered): page-based  — 50 pages/month, resets 1st of month
#   Paid (subscribed): statement-based — 1 statement consumed per conversion
#                      regardless of page count (40/120/400 per month)
def get_user_quota(user: dict) -> dict:
    tier = "registered"
    expiry_str = None
    credits_limit = REGISTERED_STMT_LIMIT
    units_used = 0
    expired = False

    if not supabase:
        print("[Quota] WARNING: Supabase client not initialized")
        return {
            "tier": "registered",
            "credits_limit": REGISTERED_STMT_LIMIT,
            "units_used": REGISTERED_STMT_LIMIT, # fail closed to prevent free bypass
            "units_remaining": 0,
            "is_statement_based": True,
            "expired": False,
            "pages_used": REGISTERED_STMT_LIMIT,
            "pages_remaining": 0,
        }

    try:
        # 1. Fetch Profile
        p_res = supabase.table("profiles").select("tier, premium_expiry_date, credits").eq("id", user["id"]).execute()
        if p_res.data and len(p_res.data) > 0:
            profile = p_res.data[0]
            tier = profile.get("tier") or "registered"
            expiry_str = profile.get("premium_expiry_date")
            raw_credits = profile.get("credits")
            if tier == "subscribed":
                credits_limit = raw_credits if raw_credits else STARTER_STMT_LIMIT
            else:
                credits_limit = REGISTERED_STMT_LIMIT
    except Exception as e:
        print(f"[Quota] Error fetching profile: {e}")

    # Check expiry for subscribed users
    if tier == "subscribed" and expiry_str:
        if expiry_str < datetime.now(timezone.utc).isoformat():
            tier = "registered"
            expired = True
            credits_limit = REGISTERED_STMT_LIMIT

    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(IST)
    month_start = now_ist.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    today_start = now_ist.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    try:
        if tier == "subscribed":
            # Paid user: count statements used this month
            c_res = supabase.table("conversions").select("id").eq("user_id", user["id"]).gte("created_at", month_start).execute()
            units_used = len(c_res.data) if c_res.data else 0
        else:
            # Free user: count sum of pages consumed today (daily 15-page limit)
            c_res = supabase.table("conversions").select("pages").eq("user_id", user["id"]).gte("created_at", today_start).execute()
            if c_res.data:
                units_used = sum(r.get("pages") or 1 for r in c_res.data)
            else:
                units_used = 0
    except Exception as e:
        print(f"[Quota] Error fetching conversions: {e}")

    units_remaining = max(0, credits_limit - units_used)

    print(f"[Quota] User {user.get('email')} ({user['id']}) - Tier: {tier}, Used: {units_used}/{credits_limit}, Remaining: {units_remaining}")

    return {
        "tier": tier,
        "credits_limit": credits_limit,
        "units_used": units_used,
        "units_remaining": units_remaining,
        "is_statement_based": (tier == "subscribed"),
        "expired": expired,
        "pages_used": units_used,
        "pages_remaining": units_remaining,
    }


# ─── Schemas ──────────────────────────────────────────────────────────────────
class TransactionSchema(BaseModel):
    date: str
    value_date: Optional[str] = ""
    description: str
    debit: Optional[str] = ""
    credit: Optional[str] = ""
    balance: Optional[str] = ""
    category: Optional[str] = "Other"
    gst: Optional[str] = ""


class DownloadRequest(BaseModel):
    transactions: List[TransactionSchema]
    filename: Optional[str] = "statement"
    format: str  # "xlsx" or "csv"


def evaluate_native_confidence(txns: List[dict], page_count: int, extracted_text: Optional[str] = None) -> bool:
    """
    Check if native parse results are good enough to use.
    Rejects native parse if transaction count is suspiciously low relative to page count or date density.
    """
    if not txns:
        return False
        
    total_txns = len(txns)
    
    # 1. Page count density check (tiered thresholds):
    # - Small/medium PDFs (≤10 pages): 2.0 txns/page (statements often have summary pages)
    # - Large PDFs (>10 pages): 3.0 txns/page (most pages should be transaction pages)
    if page_count > 2:
        min_density = 2.0 if page_count <= 10 else 3.0
        avg_txns = total_txns / page_count
        if avg_txns < min_density:
            # Rescue: if we found ≥10 txns and ≥70% have valid dates, accept anyway
            valid_date_count = sum(1 for t in txns if is_valid_date(t.get("date", "")))
            if total_txns >= 10 and valid_date_count >= total_txns * 0.70:
                print(f"[Native Confidence] Low density ({avg_txns:.1f}/page < {min_density}) but good date quality ({valid_date_count}/{total_txns}). Accepting.")
            else:
                print(f"[Native Confidence] Low txn density: {total_txns} txns for {page_count} pages ({avg_txns:.1f}/page < {min_density}). Rejecting native parse.")
                return False

    # 2. Text date count cross-check if extracted text is provided
    if extracted_text:
        date_matches = len(re.findall(
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b',
            extracted_text
        ))
        if date_matches > 20 and total_txns < date_matches * 0.4:
            print(f"[Native Confidence] Date discrepancy: {total_txns} txns vs {date_matches} dates in text. Rejecting native parse.")
            return False
        
    # 3. Check completeness (missing amounts) — allow up to 60% missing
    incomplete_count = sum(1 for t in txns if not str(t.get("debit", "")).strip() and not str(t.get("credit", "")).strip())
    if total_txns > 2 and incomplete_count > total_txns * 0.60:
        print(f"[Native Confidence] Low completeness: {incomplete_count}/{total_txns} lack amounts.")
        return False
        
    # 4. Check for valid dates
    invalid_dates = sum(1 for t in txns if not is_valid_date(t.get("date", "")))
    if total_txns > 2 and invalid_dates > total_txns * 0.30:
        print(f"[Native Confidence] Invalid dates: {invalid_dates}/{total_txns} have bad dates.")
        return False
        
    return True


# ─── Root ───────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Welcome to StatementConvert Python API", "docs": "/docs"}

# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "statement-convert-python-backend",
        "version": "2.0.0",
        "gemini_active": bool(os.getenv("GEMINI_API_KEY")),
        "supabase_active": supabase is not None,
    }


# ─── Single Convert ───────────────────────────────────────────────────────────
@app.post("/api/convert")
async def convert_statement(
    request: Request,
    file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    bank: Optional[str] = Form("auto"),
    date_format: Optional[str] = Form("DD/MM/YYYY"),
    categorize: Optional[bool] = Form(True),
    gst: Optional[bool] = Form(True),
    user: Optional[dict] = Depends(verify_user),
    x_anon_id: Optional[str] = Header(None, alias="X-Anon-Id"),
):
    # ── Validate file ──
    filename_lower = file.filename.lower()
    supported_extensions = (".pdf", ".png", ".jpg", ".jpeg", ".webp")
    if not filename_lower.endswith(supported_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, PNG, JPG, or WEBP.")

    is_image = filename_lower.endswith((".png", ".jpg", ".jpeg", ".webp"))
    mime_type = "application/pdf"
    if filename_lower.endswith(".png"):
        mime_type = "image/png"
    elif filename_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif filename_lower.endswith(".webp"):
        mime_type = "image/webp"

    temp_path = os.path.join(tempfile.gettempdir(), f"sc_{os.urandom(6).hex()}_{file.filename}")

    try:
        # Save to temp (streaming, no full RAM load)
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # File size check
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB."
            )

        is_text_based = False
        working_path = temp_path
        is_temp_decrypted = False

        if is_image:
            page_count = 1
        else:
            # ── Check & Decrypt PDF if password protected ──
            working_path, is_temp_decrypted, is_pwd_invalid = await asyncio.to_thread(
                decrypt_pdf_if_needed, temp_path, password
            )

            if is_pwd_invalid:
                origin = request.headers.get("origin", "*")
                msg = "Incorrect password for this PDF." if password else "This PDF is password protected. Please enter the password."
                return JSONResponse(
                    status_code=401,
                    content={"error": "password_required", "message": msg},
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Allow-Headers": "*",
                        "Access-Control-Allow-Methods": "*",
                    }
                )

            # ── Extract page count & basic text check ──
            try:
                page_count, is_text_based = await asyncio.to_thread(check_pdf_basic, working_path, None)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Could not read PDF: {e}")

            if page_count == 0:
                raise HTTPException(status_code=422, detail="The PDF statement is empty.")

        # ── Quota check ──
        if not user:
            if not x_anon_id:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required. Please sign up or sign in to convert bank statements."
                )
            
            if supabase:
                IST = timezone(timedelta(hours=5, minutes=30))
                today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                
                res = await asyncio.to_thread(
                    lambda: supabase.table("conversions").select("id").eq("user_id", x_anon_id).gte("created_at", today_start).execute()
                )
                anon_count = len(res.data) if res.data else 0
                
                if anon_count >= ANON_STMT_LIMIT:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Free daily limit reached. You have used your {ANON_STMT_LIMIT} free statement for today. Please sign up for more conversions."
                    )
            
            user_id_for_log = x_anon_id
        else:
            user_id_for_log = user["id"]
            quota = await asyncio.to_thread(get_user_quota, user)

            if quota["expired"]:
                print(f"User {user['id']} subscription expired — treating as registered")

            if quota["units_remaining"] <= 0:
                if quota["tier"] == "subscribed":
                    raise HTTPException(
                        status_code=403,
                        detail=f"Monthly statement limit reached. You have used all {quota['credits_limit']} statements this month. Wait for next month or upgrade your plan."
                    )
                else:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Daily free limit reached. You have used all {quota['credits_limit']} free pages today. Upgrade your plan to convert this statement."
                    )

            if quota["tier"] != "subscribed" and page_count > quota["units_remaining"]:
                raise HTTPException(
                    status_code=403,
                    detail=f"Daily limit exceeded. This statement has {page_count} pages, but you only have {quota['units_remaining']} free pages left today. Upgrade your plan to convert this statement."
                )

        # ── Parse natively (Primary, Super Fast) → Gemini (Fallback) ──
        raw_txns = []
        gemini_used_for_extraction = False
        _cached_text = None  # Cache extracted text to avoid redundant calls

        if not is_image and is_text_based:
            print(f"[Native] Parsing: {file.filename} ({page_count} pages)")
            native_success = False
            low_confidence_txns = []
            
            try:
                raw_txns = await asyncio.to_thread(parse_pdf_natively, working_path, None)
                if raw_txns:
                    print(f"[Native] [OK] {len(raw_txns)} transactions extracted natively")
                    if evaluate_native_confidence(raw_txns, page_count):
                        native_success = True
                    else:
                        print(f"[Native] [WARN] Low confidence natively. Saving as fallback.")
                        low_confidence_txns = raw_txns
                        raw_txns = []
            except Exception as e:
                print(f"[Native] [ERROR] Failed: {e} — trying line-by-line regex fallback")

            if not raw_txns and not native_success:
                print(f"[Native] Trying regex line-by-line parser fallback: {file.filename}")
                try:
                    _cached_text = await asyncio.to_thread(extract_full_text, working_path, None)
                    raw_txns = parse_line_by_line(_cached_text)
                    if raw_txns:
                        print(f"[Native] [OK] {len(raw_txns)} transactions extracted via regex line fallback")
                        if evaluate_native_confidence(raw_txns, page_count):
                            native_success = True
                        else:
                            print(f"[Native] [WARN] Low confidence line-by-line. Saving as fallback.")
                            if not low_confidence_txns:
                                low_confidence_txns = raw_txns
                            raw_txns = []
                except Exception as e:
                    print(f"[Native] Regex line parser failed: {e} — trying Gemini fallback")

            # ── Gemini Fallback — choose strategy based on page count ──
            if not raw_txns:
                if page_count >= PDF_CHUNK_THRESHOLD:
                    # Large PDFs (15+ pages): chunked parallel processing to avoid timeouts
                    print(f"[Gemini Chunked] PDF ({page_count} pages >= {PDF_CHUNK_THRESHOLD}) — using chunked processing: {file.filename}")
                    try:
                        raw_txns, _g_calls = await asyncio.to_thread(
                            parse_large_pdf_chunked,
                            working_path,
                            mime_type,
                            password=None,
                            categorize=categorize,
                            gst=gst
                        )
                        if raw_txns:
                            gemini_used_for_extraction = True
                            print(f"[Gemini Chunked] [OK] {len(raw_txns)} transactions extracted via chunked Gemini")
                    except Exception as e:
                        print(f"[Gemini Chunked] [ERROR] Chunked processing failed: {e}")
                        err_msg = str(e).lower()
                        if any(kw in err_msg for kw in ["dunning", "billing", "permission_denied", "403"]):
                            if not low_confidence_txns:
                                raise HTTPException(
                                    status_code=402,
                                    detail="Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status or update your API key."
                                )
                        if any(kw in err_msg for kw in ["429", "quota", "resource_exhausted", "limit"]):
                            if not low_confidence_txns:
                                raise HTTPException(
                                    status_code=429,
                                    detail="Gemini AI API rate limit or daily quota exceeded. Please upgrade to a paid API key or configure billing in Google AI Studio."
                                )
                elif page_count >= 5:
                    # Medium PDFs (5-14 pages): go directly to File API (faster, single call)
                    print(f"[Gemini File] PDF ({page_count} pages) — using File API directly: {file.filename}")
                    try:
                        raw_txns, _g_calls = await asyncio.to_thread(
                            parse_file_directly_with_gemini,
                            working_path,
                            mime_type,
                            categorize=categorize,
                            gst=gst
                        )
                        if raw_txns:
                            gemini_used_for_extraction = True
                            print(f"[Gemini File] [OK] {len(raw_txns)} transactions extracted via File API")
                    except Exception as e:
                        print(f"[Gemini File] [ERROR] File API failed: {e}")
                        err_msg = str(e).lower()
                        if any(kw in err_msg for kw in ["dunning", "billing", "permission_denied", "403"]):
                            if not low_confidence_txns:
                                raise HTTPException(
                                    status_code=402,
                                    detail="Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status or update your API key."
                                )
                        if any(kw in err_msg for kw in ["429", "quota", "resource_exhausted", "limit"]):
                            if not low_confidence_txns:
                                raise HTTPException(
                                    status_code=429,
                                    detail="Gemini AI API rate limit or daily quota exceeded. Please upgrade to a paid API key or configure billing in Google AI Studio."
                                )
                else:
                    # Small PDFs (<5 pages): try text chunks first (cheaper)
                    print(f"[Gemini] Fallback text parsing starting: {file.filename}")
                    try:
                        # Reuse cached text if already extracted
                        if _cached_text is None:
                            _cached_text = await asyncio.to_thread(extract_full_text, working_path, None)
                        raw_txns, _g_calls = await asyncio.to_thread(parse_with_gemini, _cached_text, categorize=categorize, gst=gst)
                        if raw_txns:
                            gemini_used_for_extraction = True
                            print(f"[Gemini] [OK] {len(raw_txns)} transactions extracted via text fallback")
                    except Exception as e:
                        print(f"[Gemini] [ERROR] Text fallback failed: {e}")
                        err_msg = str(e).lower()
                        if any(kw in err_msg for kw in ["dunning", "billing", "permission_denied", "403"]):
                            if not low_confidence_txns:
                                raise HTTPException(
                                    status_code=402,
                                    detail="Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status or update your API key."
                                )
                        if any(kw in err_msg for kw in ["429", "quota", "resource_exhausted", "limit"]):
                            if not low_confidence_txns:
                                raise HTTPException(
                                    status_code=429,
                                    detail="Gemini AI API rate limit or daily quota exceeded. Please upgrade to a paid API key or configure billing in Google AI Studio."
                                )

                    # Small PDF File API fallback if text chunks also failed
                    if not raw_txns:
                        print(f"[Gemini File] Final fallback — uploading full PDF: {file.filename}")
                        try:
                            raw_txns, _g_calls = await asyncio.to_thread(
                                parse_file_directly_with_gemini,
                                working_path,
                                mime_type,
                                categorize=categorize,
                                gst=gst
                            )
                            if raw_txns:
                                gemini_used_for_extraction = True
                                print(f"[Gemini File] [OK] {len(raw_txns)} transactions extracted via File API fallback")
                        except Exception as e:
                            print(f"[Gemini File] [ERROR] File API fallback also failed: {e}")
                            err_msg = str(e).lower()
                            if any(kw in err_msg for kw in ["dunning", "billing", "permission_denied", "403"]):
                                if not low_confidence_txns:
                                    raise HTTPException(
                                        status_code=402,
                                        detail="Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status or update your API key."
                                    )
                            if any(kw in err_msg for kw in ["429", "quota", "resource_exhausted", "limit"]):
                                if not low_confidence_txns:
                                    raise HTTPException(
                                        status_code=429,
                                        detail="Gemini AI API rate limit or daily quota exceeded. Please upgrade to a paid API key or configure billing in Google AI Studio."
                                    )

            # ── Rescue low confidence native txns if all Gemini paths failed ──
            if not raw_txns and low_confidence_txns:
                print(f"[Rescue] Using {len(low_confidence_txns)} low-confidence native txns (all Gemini paths failed)")
                raw_txns = low_confidence_txns

        else:
            # Scanned PDF or Image screenshot: parse directly via Files API multimodal
            if not is_image and page_count >= PDF_CHUNK_THRESHOLD:
                print(f"[Gemini Chunked Scanned] Scanned PDF ({page_count} pages >= {PDF_CHUNK_THRESHOLD}) — using chunked processing: {file.filename}")
                try:
                    raw_txns, _g_calls = await asyncio.to_thread(
                        parse_large_pdf_chunked,
                        working_path,
                        mime_type,
                        password=None,
                        categorize=categorize,
                        gst=gst
                    )
                    if raw_txns:
                        gemini_used_for_extraction = True
                        print(f"[Gemini Chunked Scanned] [OK] {len(raw_txns)} transactions extracted via chunked Gemini")
                except Exception as e:
                    print(f"[Gemini Chunked Scanned] [ERROR] Chunked scanned failed: {e}")
                    err_msg = str(e).lower()
                    if any(kw in err_msg for kw in ["dunning", "billing", "permission_denied", "403"]):
                        raise HTTPException(
                            status_code=402,
                            detail="Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status or update your API key."
                        )
                    if any(kw in err_msg for kw in ["429", "quota", "resource_exhausted", "limit"]):
                        raise HTTPException(
                            status_code=429,
                            detail="Gemini AI API rate limit or daily quota exceeded. Please upgrade to a paid API key or configure billing in Google AI Studio."
                        )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse scanned document: {str(e)}"
                    )
            else:
                print(f"[Gemini Direct] Parsing scanned PDF or Image directly: {file.filename}")
                try:
                    raw_txns, _g_calls = await asyncio.to_thread(
                        parse_file_directly_with_gemini,
                        working_path,
                        mime_type,
                        categorize=categorize,
                        gst=gst
                    )
                    if raw_txns:
                        gemini_used_for_extraction = True
                        print(f"[Gemini Direct] [OK] {len(raw_txns)} transactions extracted directly")
                except Exception as e:
                    print(f"[Gemini Direct] [ERROR] Direct parsing failed: {e}")
                    err_msg = str(e).lower()
                    if any(kw in err_msg for kw in ["dunning", "billing", "permission_denied", "403"]):
                        raise HTTPException(
                            status_code=402,
                            detail="Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status or update your API key."
                        )
                    if any(kw in err_msg for kw in ["429", "quota", "resource_exhausted", "limit"]):
                        raise HTTPException(
                            status_code=429,
                            detail="Gemini AI API rate limit or daily quota exceeded. Please upgrade to a paid API key or configure billing in Google AI Studio."
                        )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse document: {str(e)}"
                    )

        if not raw_txns:
            err_detail = "No transactions detected in this statement. If the PDF is password-protected, please check that the password is correct." if password else "No transactions detected in this document. Please verify the statement file or select the bank manually."
            raise HTTPException(
                status_code=422,
                detail=err_detail
            )

        cleaned = clean_and_format_transactions(raw_txns, date_format=date_format)

        # ── AI Categorization & GST Tagging ──
        if not gemini_used_for_extraction and (categorize or gst):
            print(f"[Gemini] Categorizing and tagging GST with Gemini 2.0 Flash for {len(cleaned)} transactions...")
            try:
                cleaned, _g_calls = await asyncio.to_thread(categorize_and_tag_with_gemini, cleaned, categorize=categorize, gst=gst)
                print(f"[Gemini] [OK] AI Categorization/GST completed")
            except Exception as e:
                print(f"[Gemini] [ERROR] AI Categorization failed: {e} — using local rule results")

        # Log anonymous conversion
        if not user and supabase and x_anon_id:
            try:
                await asyncio.to_thread(
                    lambda: supabase.table("conversions").insert({
                        "user_id": x_anon_id,
                        "file_name": file.filename,
                        "bank": bank if bank != "auto" else None,
                        "pages": page_count,
                        "format": "Excel",
                        "status": "done",
                        "credits": page_count,
                        "transactions_count": len(cleaned)
                    }).execute()
                )
            except Exception as e:
                print(f"Failed to log anon conversion: {e}")

        # Auto-detect bank and account info
        detected_bank_name = None
        account_holder_name = None
        account_number = None
        try:
            sample_text = _cached_text if _cached_text else await asyncio.to_thread(extract_full_text, working_path, None)
            if sample_text:
                detected_bank_name = detect_bank_name(sample_text)
                acc_info = detect_account_details(sample_text)
                account_holder_name = acc_info.get("account_holder_name")
                account_number = acc_info.get("account_number")
        except Exception:
            pass

        # ── 99.9% Mathematical Reconciliation & Self-Healing ──
        cleaned, audit = reconcile_and_heal_transactions(cleaned)

        return {
            "success": True,
            "filename": file.filename,
            "pages": page_count,
            "bank_name": detected_bank_name or (bank if bank != "auto" else None),
            "account_holder_name": account_holder_name,
            "account_number": account_number,
            "transactions": cleaned,
            "transactions_count": len(cleaned),
            "reconciliation": audit,
            "user_id": user_id_for_log,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if is_temp_decrypted and working_path and os.path.exists(working_path):
            os.remove(working_path)


# ─── Batch Convert ────────────────────────────────────────────────────────────
@app.post("/api/convert/batch")
async def convert_batch(
    files: List[UploadFile] = File(...),
    passwords: Optional[str] = Form(None),  # JSON: {"0": "pwd", "1": "pwd2"}
    bank: Optional[str] = Form("auto"),
    date_format: Optional[str] = Form("DD/MM/YYYY"),
    categorize: Optional[bool] = Form(True),
    gst: Optional[bool] = Form(True),
    user: Optional[dict] = Depends(verify_user),
):
    # ── Auth check ──
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign up or sign in to use bulk conversion."
        )

    # ── File count limit ──
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_BATCH_FILES} files per batch."
        )

    # ── Tier check — ONE query ──
    quota = get_user_quota(user)
    if quota["tier"] != "subscribed":
        raise HTTPException(
            status_code=403,
            detail="Bulk conversion is a premium feature. Upgrade to Starter (₹999), Growth (₹1999), or Pro (₹3499) plan."
        )

    if quota["units_remaining"] <= 0:
        unit_label = "statements" if quota["is_statement_based"] else "pages"
        raise HTTPException(
            status_code=403,
            detail=f"Monthly quota exhausted. You have used all {quota['credits_limit']} {unit_label} this month."
        )

    # Parse password map
    pwd_map: dict = {}
    if passwords:
        try:
            pwd_map = json.loads(passwords)
        except Exception:
            pass

    # ── Process each file in parallel (true async) ──
    async def process_one(idx: int, f: UploadFile):
        temp_path = os.path.join(tempfile.gettempdir(), f"sc_batch_{os.urandom(6).hex()}_{idx}_{f.filename}")
        try:
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(f.file, buf)

            # File size check
            file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return {
                    "index": idx, "filename": f.filename, "success": False,
                    "error": f"File too large ({file_size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB allowed."
                }

            pwd = pwd_map.get(str(idx)) or pwd_map.get(idx)

            filename_lower = f.filename.lower()
            supported_extensions = (".pdf", ".png", ".jpg", ".jpeg", ".webp")
            if not filename_lower.endswith(supported_extensions):
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": "Unsupported file format. Please upload PDF, PNG, JPG, or WEBP."}

            is_image = filename_lower.endswith((".png", ".jpg", ".jpeg", ".webp"))
            mime_type = "application/pdf"
            if filename_lower.endswith(".png"):
                mime_type = "image/png"
            elif filename_lower.endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif filename_lower.endswith(".webp"):
                mime_type = "image/webp"

            is_text_based = False
            if is_image:
                page_count = 1
            else:
                try:
                    page_count, is_text_based = await asyncio.to_thread(check_pdf_basic, temp_path, pwd)
                except Exception as e:
                    err = str(e).lower()
                    err_repr = repr(e).lower()
                    err_type = type(e).__name__.lower()
                    is_pwd_err = any(keyword in (err + err_repr + err_type) for keyword in ["password", "decrypt", "encrypt", "incorrect"])
                    if is_pwd_err:
                        return {"index": idx, "filename": f.filename, "success": False,
                                "error": "password_required", "message": "PDF is password protected."}
                    return {"index": idx, "filename": f.filename, "success": False, "error": str(e)}

                if page_count == 0:
                    return {"index": idx, "filename": f.filename, "success": False, "error": "Empty PDF."}

            # Per-file quota check: paid plans count statements, not pages
            # (quota was fetched once before the batch loop)
            if quota["units_remaining"] <= 0:
                unit_label = "statements" if quota["is_statement_based"] else "pages"
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": f"Monthly limit reached. No {unit_label} remaining this month."}

            # Parse natively first for extreme speed
            raw_txns = []
            gemini_used_for_extraction = False
            _cached_text = None  # Cache text extraction

            if not is_image and is_text_based:
                native_success = False
                low_confidence_txns = []
                try:
                    raw_txns = await asyncio.to_thread(parse_pdf_natively, temp_path, pwd)
                    if raw_txns and evaluate_native_confidence(raw_txns, page_count):
                        native_success = True
                    elif raw_txns:
                        low_confidence_txns = raw_txns
                        raw_txns = []
                except Exception as e:
                    print(f"[Native] Failed for {f.filename}: {e}")

                if not raw_txns and not native_success:
                    try:
                        _cached_text = await asyncio.to_thread(extract_full_text, temp_path, pwd)
                        raw_txns = parse_line_by_line(_cached_text)
                        if raw_txns and evaluate_native_confidence(raw_txns, page_count):
                            native_success = True
                        elif raw_txns:
                            if not low_confidence_txns:
                                low_confidence_txns = raw_txns
                            raw_txns = []
                    except Exception as e:
                        pass

                if not raw_txns:
                    if page_count >= PDF_CHUNK_THRESHOLD:
                        # Large PDFs (15+ pages): chunked parallel processing
                        print(f"[Gemini Chunked] Batch PDF ({page_count} pages >= {PDF_CHUNK_THRESHOLD}) — using chunked processing: {f.filename}")
                        try:
                            raw_txns, _g_calls = await asyncio.to_thread(
                                parse_large_pdf_chunked,
                                temp_path,
                                mime_type,
                                password=pwd,
                                categorize=categorize,
                                gst=gst
                            )
                            if raw_txns:
                                gemini_used_for_extraction = True
                        except Exception as e:
                            print(f"[Gemini Chunked] Batch chunked failed for {f.filename}: {e}")
                    elif page_count >= 5:
                        # Medium PDFs: File API directly
                        print(f"[Gemini File] Batch PDF ({page_count} pages) — File API: {f.filename}")
                        try:
                            raw_txns, _g_calls = await asyncio.to_thread(
                                parse_file_directly_with_gemini,
                                temp_path, mime_type,
                                categorize=categorize, gst=gst
                            )
                            if raw_txns:
                                gemini_used_for_extraction = True
                        except Exception as e:
                            print(f"[Gemini File] Batch File API failed for {f.filename}: {e}")
                    else:
                        # Small PDFs: text chunks first
                        print(f"[Gemini] Fallback text parsing for batch file: {f.filename}")
                        try:
                            if _cached_text is None:
                                _cached_text = await asyncio.to_thread(extract_full_text, temp_path, pwd)
                            raw_txns, _g_calls = await asyncio.to_thread(parse_with_gemini, _cached_text, categorize=categorize, gst=gst)
                            if raw_txns:
                                gemini_used_for_extraction = True
                        except Exception as e:
                            print(f"[Gemini] Batch text fallback failed for {f.filename}: {e}")

                # Gemini File API fallback for batch
                if not raw_txns:
                    print(f"[Gemini File] Final fallback for batch file: {f.filename}")
                    try:
                        raw_txns, _g_calls = await asyncio.to_thread(
                            parse_file_directly_with_gemini,
                            temp_path,
                            mime_type,
                            categorize=categorize,
                            gst=gst
                        )
                        if raw_txns:
                            gemini_used_for_extraction = True
                    except Exception as e:
                        print(f"[Gemini File] Batch File API fallback failed for {f.filename}: {e}")

                # Rescue low confidence native txns if all Gemini paths failed
                if not raw_txns and low_confidence_txns:
                    print(f"[Rescue] Using {len(low_confidence_txns)} low-confidence native txns for {f.filename}")
                    raw_txns = low_confidence_txns

            else:
                # Scanned PDF or Image screenshot
                if not is_image and page_count >= PDF_CHUNK_THRESHOLD:
                    print(f"[Gemini Chunked Scanned] Batch scanned PDF ({page_count} pages >= {PDF_CHUNK_THRESHOLD}) — using chunked processing: {f.filename}")
                    try:
                        raw_txns, _g_calls = await asyncio.to_thread(
                            parse_large_pdf_chunked,
                            temp_path,
                            mime_type,
                            password=pwd,
                            categorize=categorize,
                            gst=gst
                        )
                        if raw_txns:
                            gemini_used_for_extraction = True
                    except Exception as e:
                        print(f"[Gemini Chunked Scanned] Batch chunked scanned failed for {f.filename}: {e}")
                        err_msg = str(e).lower()
                        if "dunning" in err_msg or "billing" in err_msg or "permission_denied" in err_msg or "403" in err_msg:
                            return {"index": idx, "filename": f.filename, "success": False,
                                    "error": "Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status."}
                        return {"index": idx, "filename": f.filename, "success": False, "error": f"AI parsing failed: {str(e)}"}
                else:
                    print(f"[Gemini Direct] Batch processing scanned PDF or Image directly: {f.filename}")
                    try:
                        raw_txns, _g_calls = await asyncio.to_thread(
                            parse_file_directly_with_gemini,
                            temp_path,
                            mime_type,
                            categorize=categorize,
                            gst=gst
                        )
                        if raw_txns:
                            gemini_used_for_extraction = True
                    except Exception as e:
                        print(f"[Gemini Direct] Batch direct parsing failed for {f.filename}: {e}")
                        err_msg = str(e).lower()
                        if "dunning" in err_msg or "billing" in err_msg or "permission_denied" in err_msg or "403" in err_msg:
                            return {"index": idx, "filename": f.filename, "success": False,
                                    "error": "Gemini AI API Key has a billing issue. Please check your Google Cloud Billing status."}
                        return {"index": idx, "filename": f.filename, "success": False, "error": f"AI parsing failed: {str(e)}"}

            if not raw_txns:
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": "No transactions detected. Make sure it is a valid Indian bank statement."}

            cleaned = clean_and_format_transactions(raw_txns, date_format=date_format)

            # AI Categorization & GST Tagging
            if not gemini_used_for_extraction and (categorize or gst):
                try:
                    cleaned, _g_calls = await asyncio.to_thread(categorize_and_tag_with_gemini, cleaned, categorize=categorize, gst=gst)
                except Exception as e:
                    print(f"[Gemini] Batch categorization failed for {f.filename}: {e}")

            # Auto-detect bank and account info for this file
            detected_bank_name = None
            account_holder_name = None
            account_number = None
            try:
                sample_text = _cached_text if _cached_text else await asyncio.to_thread(extract_full_text, temp_path, pwd)
                if sample_text:
                    detected_bank_name = detect_bank_name(sample_text)
                    acc_info = detect_account_details(sample_text)
                    account_holder_name = acc_info.get("account_holder_name")
                    account_number = acc_info.get("account_number")
            except Exception:
                pass

            # ── 99.9% Mathematical Reconciliation & Self-Healing ──
            cleaned, audit = reconcile_and_heal_transactions(cleaned)

            return {
                "index": idx,
                "filename": f.filename,
                "success": True,
                "pages": page_count,
                "transactions": cleaned,
                "transactions_count": len(cleaned),
                "reconciliation": audit,
                "bank_name": detected_bank_name or (bank if bank != "auto" else None),
                "account_holder_name": account_holder_name,
                "account_number": account_number,
            }

        except Exception as e:
            return {"index": idx, "filename": f.filename, "success": False, "error": str(e)}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Limit concurrency to prevent Out-Of-Memory (OOM) crashes on small servers
    semaphore = asyncio.Semaphore(2)
    
    async def process_with_limit(idx, f):
        async with semaphore:
            return await process_one(idx, f)

    results = await asyncio.gather(*[process_with_limit(i, f) for i, f in enumerate(files)])

    successful = [r for r in results if r.get("success")]
    total_pages = sum(r.get("pages", 0) for r in successful)

    return {
        "results": list(results),
        "user_id": user["id"],
        "summary": {
            "total_files": len(files),
            "successful": len(successful),
            "failed": len(files) - len(successful),
            "total_pages": total_pages,
            "total_transactions": sum(r.get("transactions_count", 0) for r in successful),
        }
    }


# ─── Quota Status endpoint — frontend calls this for real-time quota display ─
@app.get("/api/quota")
async def get_quota(user: Optional[dict] = Depends(verify_user)):
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign up or sign in to view quota."
        )
    quota = get_user_quota(user)
    return {
        "tier": quota["tier"],
        "credits_limit": quota["credits_limit"],
        "units_used": quota["units_used"],
        "units_remaining": quota["units_remaining"],
        "is_statement_based": quota["is_statement_based"],
        "expired": quota["expired"],
        # Legacy aliases
        "pages_remaining": quota["pages_remaining"],
        "pages_used": quota["pages_used"],
    }


# ─── Download ─────────────────────────────────────────────────────────────────
@app.post("/api/download")
async def download_file(req: DownloadRequest):
    if not req.transactions:
        raise HTTPException(status_code=400, detail="No transactions to download.")

    temp_dir = tempfile.gettempdir()
    txns_data = [t.model_dump() for t in req.transactions]

    safe_name = "".join(
        c for c in req.filename if c.isalpha() or c.isdigit() or c in " ._-"
    ).rstrip()
    safe_name = os.path.splitext(safe_name)[0] or "statement"

    if req.format == "xlsx":
        file_path = os.path.join(temp_dir, f"{safe_name}.xlsx")
        generate_excel_file(txns_data, file_path)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename_out = f"{safe_name}.xlsx"
    elif req.format == "csv":
        file_path = os.path.join(temp_dir, f"{safe_name}.csv")
        generate_csv_file(txns_data, file_path)
        media_type = "text/csv"
        filename_out = f"{safe_name}.csv"
    else:
        raise HTTPException(status_code=400, detail="Format must be 'xlsx' or 'csv'.")

    class DeleteOnCloseFileResponse(FileResponse):
        def close(self) -> None:
            super().close()
            try:
                if os.path.exists(self.path):
                    os.remove(self.path)
            except Exception as e:
                print(f"Cleanup error {self.path}: {e}")

    return DeleteOnCloseFileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename_out,
        headers={"Access-Control-Expose-Headers": "Content-Disposition"},
    )
