import os
import shutil
import tempfile
import asyncio
import json
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List

from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from parser import (
    extract_text_from_pdf,
    parse_pdf_natively,
    parse_with_gemini,
    clean_and_format_transactions,
    generate_excel_file,
    generate_csv_file,
)
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ────────────────────────────────────────────────────────────────
ANON_PAGE_LIMIT       = 1    # anonymous: 1 page per request
REGISTERED_PAGE_LIMIT = 5    # free users: 5 pages per day
MAX_BATCH_FILES       = 20   # max files per batch request
MAX_FILE_SIZE_MB      = 25   # max single file size in MB


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
            }
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
    return None


# ─── Quota helper — fetches profile + usage in 2 queries (cached per request) ─
def get_user_quota(user_id: str) -> dict:
    """
    Returns a dict with:
      tier, credits_limit, pages_used (today for registered, month for subscribed),
      pages_remaining, expired (bool)
    All in 2 Supabase queries.
    """
    if not supabase:
        return {"tier": "registered", "credits_limit": REGISTERED_PAGE_LIMIT,
                "pages_used": 0, "pages_remaining": REGISTERED_PAGE_LIMIT, "expired": False}

    # Query 1: Profile
    profile_res = supabase.table("profiles")\
        .select("tier, premium_expiry_date, credits")\
        .eq("id", user_id).execute()

    profile = profile_res.data[0] if profile_res.data else {}
    tier = profile.get("tier", "registered")
    expiry_str = profile.get("premium_expiry_date")
    credits_limit = profile.get("credits", 500) or 500

    # Check expiry
    expired = False
    if tier == "subscribed" and expiry_str:
        if expiry_str < datetime.now(timezone.utc).isoformat():
            tier = "registered"
            expired = True

    now = datetime.now(timezone.utc)

    if tier == "subscribed":
        # Query 2a: pages this month
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        conv_res = supabase.table("conversions")\
            .select("pages")\
            .eq("user_id", user_id)\
            .gte("created_at", month_start).execute()
        pages_used = sum(c["pages"] for c in (conv_res.data or []))
        pages_remaining = max(0, credits_limit - pages_used)
    else:
        # Query 2b: pages today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        conv_res = supabase.table("conversions")\
            .select("pages")\
            .eq("user_id", user_id)\
            .gte("created_at", today_start).execute()
        pages_used = sum(c["pages"] for c in (conv_res.data or []))
        credits_limit = REGISTERED_PAGE_LIMIT
        pages_remaining = max(0, REGISTERED_PAGE_LIMIT - pages_used)

    return {
        "tier": tier,
        "credits_limit": credits_limit,
        "pages_used": pages_used,
        "pages_remaining": pages_remaining,
        "expired": expired,
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
    file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    bank: Optional[str] = Form("auto"),
    date_format: Optional[str] = Form("DD/MM/YYYY"),
    categorize: Optional[bool] = Form(True),
    gst: Optional[bool] = Form(True),
    user: Optional[dict] = Depends(verify_user),
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

        # ── Extract text + page count ──
        try:
            text, page_count = extract_text_from_pdf(temp_path, password=password)
        except Exception as e:
            err = str(e).lower()
            if "password" in err or "decrypt" in err or "encrypted" in err:
                return JSONResponse(
                    status_code=401,
                    content={"error": "password_required", "message": "This PDF is password protected. Please enter the password."}
                )
            raise HTTPException(status_code=500, detail=f"Could not read PDF: {e}")

        if not text or len(text.strip()) < 20:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. It may be a scanned image. Please upload a text-based bank statement."
            )

        # ── Quota check ──
        if not user:
            # Anonymous — max 1 page
            if page_count > ANON_PAGE_LIMIT:
                raise HTTPException(
                    status_code=403,
                    detail=f"Anonymous users can only convert {ANON_PAGE_LIMIT} page. Sign up for free to convert up to 5 pages/day!"
                )
        else:
            quota = get_user_quota(user["id"])

            if quota["expired"]:
                print(f"User {user['id']} subscription expired — treating as registered")

            if quota["pages_remaining"] <= 0:
                period = "this month" if quota["tier"] == "subscribed" else "today"
                raise HTTPException(
                    status_code=403,
                    detail=f"Quota exceeded. You have used all {quota['credits_limit']} pages {period}. {'Upgrade your plan' if quota['tier'] != 'subscribed' else 'Wait for next month or upgrade plan'} for more."
                )

            if page_count > quota["pages_remaining"]:
                period = "this month" if quota["tier"] == "subscribed" else "today"
                raise HTTPException(
                    status_code=403,
                    detail=f"Not enough quota. This document has {page_count} pages but you only have {quota['pages_remaining']} pages remaining {period}."
                )

        # ── Parse with Gemini (primary) → native (fallback) ──
        raw_txns = []
        print(f"[Gemini] Parsing: {file.filename} ({page_count} pages)")

        try:
            raw_txns = parse_with_gemini(text[:600000], categorize=categorize, gst=gst)
            if raw_txns:
                print(f"[Gemini] ✅ {len(raw_txns)} transactions extracted")
        except Exception as e:
            print(f"[Gemini] ❌ Failed: {e} — trying native parser")

        if not raw_txns:
            print(f"[Native] Parsing: {file.filename}")
            try:
                raw_txns = parse_pdf_natively(temp_path, password=password)
                if raw_txns:
                    print(f"[Native] ✅ {len(raw_txns)} transactions extracted")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"All parsers failed: {e}")

        if not raw_txns:
            raise HTTPException(
                status_code=422,
                detail="No transactions detected in this document. Make sure it is a valid Indian bank statement."
            )

        cleaned = clean_and_format_transactions(raw_txns, date_format=date_format)

        return {
            "success": True,
            "filename": file.filename,
            "pages": page_count,
            "transactions": cleaned,
            "transactions_count": len(cleaned),
            "user_id": user["id"] if user else None,
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
            status_code=403,
            detail="Please log in to use bulk conversion."
        )

    # ── File count limit ──
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_BATCH_FILES} files per batch."
        )

    # ── Tier check — ONE query ──
    quota = get_user_quota(user["id"])
    if quota["tier"] != "subscribed":
        raise HTTPException(
            status_code=403,
            detail="Bulk conversion is a premium feature. Upgrade to Starter (₹999), Professional (₹1999), or Business (₹3499) plan."
        )

    if quota["pages_remaining"] <= 0:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly quota exhausted. You have used all {quota['credits_limit']} pages this month."
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
                text, page_count = extract_text_from_pdf(temp_path, password=pwd)
            except Exception as e:
                err = str(e).lower()
                if "password" in err or "decrypt" in err:
                    return {"index": idx, "filename": f.filename, "success": False,
                            "error": "password_required", "message": "PDF is password protected."}
                return {"index": idx, "filename": f.filename, "success": False, "error": str(e)}

            if not text or len(text.strip()) < 20:
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": "Could not extract text. May be a scanned image."}

            # Per-file quota check using cached quota (pages_remaining from single query)
            if page_count > quota["pages_remaining"]:
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": f"Not enough quota for this file ({page_count} pages, {quota['pages_remaining']} remaining)."}

            # Parse
            raw_txns = []
            try:
                raw_txns = parse_with_gemini(text[:600000], categorize=categorize, gst=gst)
            except Exception as e:
                print(f"[Gemini] Failed for {f.filename}: {e}")

            if not raw_txns:
                try:
                    raw_txns = parse_pdf_natively(temp_path, password=pwd)
                except Exception as e:
                    return {"index": idx, "filename": f.filename, "success": False,
                            "error": f"All parsers failed: {e}"}

            if not raw_txns:
                return {"index": idx, "filename": f.filename, "success": False,
                        "error": "No transactions detected."}

            cleaned = clean_and_format_transactions(raw_txns, date_format=date_format)
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

    # All files processed in parallel — no sequential waits
    results = await asyncio.gather(*[process_one(i, f) for i, f in enumerate(files)])

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
        return {
            "tier": "anonymous",
            "pages_remaining": ANON_PAGE_LIMIT,
            "credits_limit": ANON_PAGE_LIMIT,
            "pages_used": 0,
        }
    quota = get_user_quota(user["id"])
    return {
        "tier": quota["tier"],
        "pages_remaining": quota["pages_remaining"],
        "credits_limit": quota["credits_limit"],
        "pages_used": quota["pages_used"],
        "expired": quota["expired"],
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
