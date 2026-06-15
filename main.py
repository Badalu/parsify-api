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
    extract_full_text,
    parse_pdf_natively,
    parse_with_gemini,
    parse_line_by_line,
    clean_and_format_transactions,
    generate_excel_file,
    generate_csv_file,
    categorize_and_tag_with_gemini,
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
ANON_STMT_LIMIT            = 1    # anonymous users: 1 statement per day
REGISTERED_STMT_LIMIT      = 2    # free users: 2 statements per day
STARTER_STMT_LIMIT         = 40   # starter plan: 40 statements/month
GROWTH_STMT_LIMIT          = 120  # growth plan: 120 statements/month
PRO_STMT_LIMIT             = 400  # pro plan: 400 statements/month
MAX_BATCH_FILES            = 20   # max files per batch request
MAX_FILE_SIZE_MB           = 25   # max single file size in MB


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
    if not SUPABASE_URL:
        return {
            "tier": "registered",
            "credits_limit": REGISTERED_STMT_LIMIT,
            "units_used": 0,
            "units_remaining": REGISTERED_STMT_LIMIT,
            "is_statement_based": True,
            "expired": False,
            # Legacy aliases kept for backward-compat:
            "pages_used": 0,
            "pages_remaining": REGISTERED_STMT_LIMIT,
        }

    headers = {
        "apikey": os.getenv("SUPABASE_PUBLISHABLE_KEY") or SUPABASE_KEY,
        "Authorization": f"Bearer {user['token']}",
    }

    tier = "registered"
    expiry_str = None
    credits_limit = REGISTERED_STMT_LIMIT  # will be overridden for subscribed

    try:
        # Query 1: Profile
        url = f"{SUPABASE_URL}/rest/v1/profiles?select=tier,premium_expiry_date,credits&id=eq.{user['id']}"
        res = httpx.get(url, headers=headers)
        if res.status_code == 200 and res.json():
            profile = res.json()[0]
            tier = profile.get("tier", "registered")
            expiry_str = profile.get("premium_expiry_date")
            # credits column stores statement-limit for paid, page-limit for free
            raw_credits = profile.get("credits")
            if tier == "subscribed":
                credits_limit = raw_credits if raw_credits else STARTER_STMT_LIMIT
            else:
                credits_limit = REGISTERED_STMT_LIMIT
    except Exception as e:
        print(f"Error fetching profile: {e}")

    # Check expiry
    expired = False
    if tier == "subscribed" and expiry_str:
        if expiry_str < datetime.now(timezone.utc).isoformat():
            tier = "registered"
            expired = True
            credits_limit = REGISTERED_STMT_LIMIT

    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(IST)
    month_start = now_ist.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    today_start = now_ist.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    # ── Usage counting depends on tier ──
    is_statement_based = True
    units_used = 0

    try:
        if tier == "subscribed":
            # Count number of conversions (1 statement = 1 unit) THIS MONTH
            c_url = (
                f"{SUPABASE_URL}/rest/v1/conversions?select=id"
                f"&user_id=eq.{user['id']}&created_at=gte.{month_start}"
            )
            c_res = httpx.get(c_url, headers=headers)
            if c_res.status_code == 200:
                units_used = len(c_res.json())
        else:
            # Free user: count statements consumed TODAY
            c_url = (
                f"{SUPABASE_URL}/rest/v1/conversions?select=id"
                f"&user_id=eq.{user['id']}&created_at=gte.{today_start}"
            )
            c_res = httpx.get(c_url, headers=headers)
            if c_res.status_code == 200:
                units_used = len(c_res.json())
    except Exception as e:
        print(f"Error fetching conversions: {e}")

    units_remaining = max(0, credits_limit - units_used)

    return {
        "tier": tier,
        "credits_limit": credits_limit,
        "units_used": units_used,
        "units_remaining": units_remaining,
        "is_statement_based": is_statement_based,
        "expired": expired,
        # Legacy aliases (pages) for backward compat with frontend
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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

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

        # ── Extract page count & basic text check ──
        try:
            page_count, is_text_based = await asyncio.to_thread(check_pdf_basic, temp_path, password)
        except Exception as e:
            err = str(e).lower()
            err_repr = repr(e).lower()
            err_type = type(e).__name__.lower()
            is_pwd_err = any(keyword in (err + err_repr + err_type) for keyword in ["password", "decrypt", "encrypt", "incorrect"])
            if is_pwd_err:
                origin = request.headers.get("origin", "*")
                return JSONResponse(
                    status_code=401,
                    content={"error": "password_required", "message": "This PDF is password protected. Please enter the password."},
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Allow-Headers": "*",
                        "Access-Control-Allow-Methods": "*",
                    }
                )
            raise HTTPException(status_code=500, detail=f"Could not read PDF: {e}")

        if not is_text_based or page_count == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. It may be a scanned image. Please upload a text-based bank statement."
            )

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
                        detail=f"Free daily quota exhausted. You have used your {REGISTERED_STMT_LIMIT} statements today. Register/upgrade to get more conversions."
                    )

        # ── Parse natively (Primary, Super Fast) → Gemini (Fallback) ──
        raw_txns = []
        gemini_used_for_extraction = False
        print(f"[Native] Parsing: {file.filename} ({page_count} pages)")

        try:
            raw_txns = await asyncio.to_thread(parse_pdf_natively, temp_path, password)
            if raw_txns:
                print(f"[Native] ✅ {len(raw_txns)} transactions extracted")
        except Exception as e:
            print(f"[Native] ❌ Failed: {e} — trying line-by-line regex fallback")

        if not raw_txns:
            print(f"[Native] Trying regex line-by-line parser fallback: {file.filename}")
            try:
                text = await asyncio.to_thread(extract_full_text, temp_path, password)
                raw_txns = parse_line_by_line(text)
                if raw_txns:
                    print(f"[Native] ✅ {len(raw_txns)} transactions extracted via regex line fallback")
            except Exception as e:
                print(f"[Native] Regex line parser failed: {e} — trying Gemini fallback")

        # ── Gemini Fallback ──
        if not raw_txns:
            print(f"[Gemini] Fallback parsing starting: {file.filename}")
            try:
                text = await asyncio.to_thread(extract_full_text, temp_path, password)
                raw_txns = await asyncio.to_thread(parse_with_gemini, text, categorize=categorize, gst=gst)
                if raw_txns:
                    gemini_used_for_extraction = True
                    print(f"[Gemini] ✅ {len(raw_txns)} transactions extracted via fallback")
            except Exception as e:
                print(f"[Gemini] ❌ Fallback parsing failed: {e}")

        if not raw_txns:
            raise HTTPException(
                status_code=422,
                detail="No transactions detected in this document. Make sure it is a valid Indian bank statement."
            )

        cleaned = clean_and_format_transactions(raw_txns, date_format=date_format)

        # ── AI Categorization & GST Tagging ──
        if not gemini_used_for_extraction and (categorize or gst):
            print(f"[Gemini] Categorizing and tagging GST with Gemini 2.0 Flash for {len(cleaned)} transactions...")
            try:
                cleaned = await asyncio.to_thread(categorize_and_tag_with_gemini, cleaned, categorize=categorize, gst=gst)
                print(f"[Gemini] ✅ AI Categorization/GST completed")
            except Exception as e:
                print(f"[Gemini] ❌ AI Categorization failed: {e} — using local rule results")

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

        return {
            "success": True,
            "filename": file.filename,
            "pages": page_count,
            "transactions": cleaned,
            "transactions_count": len(cleaned),
            "user_id": user_id_for_log,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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
            detail="Bulk conversion is a premium feature. Upgrade to Starter (₹999), Growth (₹1999), or Pro (₹3400) plan."
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

            if not is_text_based or page_count == 0:
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": "Could not extract text. May be a scanned image."}

            # Per-file quota check: paid plans count statements, not pages
            # (quota was fetched once before the batch loop)
            if quota["units_remaining"] <= 0:
                unit_label = "statements" if quota["is_statement_based"] else "pages"
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": f"Monthly limit reached. No {unit_label} remaining this month."}

            # Parse natively first for extreme speed
            raw_txns = []
            gemini_used_for_extraction = False
            try:
                raw_txns = await asyncio.to_thread(parse_pdf_natively, temp_path, pwd)
            except Exception as e:
                print(f"[Native] Failed for {f.filename}: {e}")

            if not raw_txns:
                try:
                    text = await asyncio.to_thread(extract_full_text, temp_path, pwd)
                    raw_txns = parse_line_by_line(text)
                except Exception as e:
                    pass

            if not raw_txns:
                print(f"[Gemini] Fallback parsing starting for batch file: {f.filename}")
                try:
                    text = await asyncio.to_thread(extract_full_text, temp_path, pwd)
                    raw_txns = await asyncio.to_thread(parse_with_gemini, text, categorize=categorize, gst=gst)
                    if raw_txns:
                        gemini_used_for_extraction = True
                except Exception as e:
                    print(f"[Gemini] Batch fallback failed for {f.filename}: {e}")

            if not raw_txns:
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": "No transactions detected."}

            cleaned = clean_and_format_transactions(raw_txns, date_format=date_format)

            # AI Categorization & GST Tagging
            if not gemini_used_for_extraction and (categorize or gst):
                try:
                    cleaned = await asyncio.to_thread(categorize_and_tag_with_gemini, cleaned, categorize=categorize, gst=gst)
                except Exception as e:
                    print(f"[Gemini] Batch categorization failed for {f.filename}: {e}")

            return {
                "index": idx,
                "filename": f.filename,
                "success": True,
                "pages": page_count,
                "transactions": cleaned,
                "transactions_count": len(cleaned),
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
