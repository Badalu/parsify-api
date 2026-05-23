import os
import shutil
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv

# Load variables from the directory where main.py is located
from pathlib import Path
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from parser import extract_text_from_pdf, parse_pdf_natively, parse_with_gemini, clean_and_format_transactions, generate_excel_file, generate_csv_file
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Failed to create Supabase client: {e}")

app = FastAPI(title="StatementConvert Python API", version="1.0.0")

# Set up CORS to allow requests from the Next.js and TanStack frontends
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

# Helper to verify auth token using Supabase client
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
                "role": user_res.user.role
            }
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")
    return None

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

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "statement-convert-python-backend",
        "gemini_active": bool(os.getenv("GEMINI_API_KEY")),
        "supabase_active": bool(supabase is not None)
    }

@app.post("/api/convert")
async def convert_statement(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    bank: Optional[str] = Form("auto"),
    date_format: Optional[str] = Form("DD/MM/YYYY"),
    categorize: Optional[bool] = Form(True),
    gst: Optional[bool] = Form(True),
    user: Optional[dict] = Depends(verify_user)
):
    # Verify file is a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF statements are supported.")

    # Save to a temporary file for parsing
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"upload_{file.filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract text and check pages
        try:
            text, page_count = extract_text_from_pdf(temp_file_path, password=password)
        except Exception as e:
            # Handle password required / wrong password errors
            err_msg = str(e).lower()
            if "password" in err_msg or "decrypt" in err_msg:
                return JSONResponse(
                    status_code=401, 
                    content={"error": "password_required", "message": "This PDF is encrypted. Enter a password to open it."}
                )
            raise HTTPException(status_code=500, detail=f"Failed to read PDF file: {e}")

        # Basic text content checks
        if not text or len(text.strip()) < 20:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. It may be a scanned image or empty. Please upload a text-based PDF statement."
            )

        # Apply basic quotas
        if not user:
            # Anonymous check (Max 1 page limit for anonymous)
            if page_count > 1:
                raise HTTPException(
                    status_code=403,
                    detail="Anonymous conversions are limited to 1 page. Please sign up for a free account to convert longer statements!"
                )
        else:
            if supabase:
                try:
                    profile_res = supabase.table("profiles").select("tier, premium_expiry_date, credits").eq("id", user["id"]).execute()
                    tier = profile_res.data[0].get("tier", "registered") if profile_res.data else "registered"
                    expiry_date_str = profile_res.data[0].get("premium_expiry_date") if profile_res.data else None
                    credits_limit = profile_res.data[0].get("credits", 500) if profile_res.data else 500
                    
                    if tier == "subscribed" and expiry_date_str:
                        from datetime import datetime, timezone
                        now_iso = datetime.now(timezone.utc).isoformat()
                        if expiry_date_str < now_iso:
                            tier = "registered"
                    
                    if tier == "subscribed":
                        from datetime import datetime, timezone
                        today = datetime.now(timezone.utc)
                        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
                        conv_res = supabase.table("conversions").select("pages").eq("user_id", user["id"]).gte("created_at", month_start).execute()
                        pages_this_month = sum(c["pages"] for c in conv_res.data) if conv_res.data else 0
                        
                        if pages_this_month + page_count > credits_limit:
                            raise HTTPException(
                                status_code=403,
                                detail=f"Monthly limit exceeded. You have {max(0, credits_limit - pages_this_month)} pages left, but this document has {page_count} pages."
                            )
                    else:
                        from datetime import datetime, timezone
                        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                        conv_res = supabase.table("conversions").select("pages").eq("user_id", user["id"]).gte("created_at", today_start).execute()
                        pages_today = sum(c["pages"] for c in conv_res.data) if conv_res.data else 0
                        
                        if pages_today + page_count > 5:
                            raise HTTPException(
                                status_code=403,
                                detail=f"Daily limit exceeded. You have {max(0, 5 - pages_today)} pages left today, but this document has {page_count} pages. Upgrade to Pro for more."
                            )
                except HTTPException:
                    raise
                except Exception as e:
                    print(f"Failed to check quota: {e}")
        
        # 2. Use Gemini AI as primary parser for 100% accuracy across all banks
        raw_txns = []
        print(f"Using Gemini AI for maximum accuracy: {file.filename}")
        try:
            limited_text = text[:600000]
            raw_txns = parse_with_gemini(limited_text, categorize=categorize, gst=gst)
            if raw_txns:
                print(f"Gemini AI extraction succeeded! Extracted {len(raw_txns)} transactions.")
        except Exception as e:
            print(f"Gemini AI failed: {e}. Falling back to native parser...")

        # Fallback to native parser only if Gemini fails (API unavailable, quota, etc.)
        if not raw_txns:
            print(f"Gemini unavailable — using native parser as fallback for: {file.filename}")
            try:
                raw_txns = parse_pdf_natively(temp_file_path, password=password)
                if raw_txns:
                    print(f"Native fallback succeeded! Extracted {len(raw_txns)} transactions.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"All parsers failed: {e}")

        if not raw_txns:
            raise HTTPException(
                status_code=422,
                detail="No transaction entries could be detected in this bank statement. Try checking your PDF or choose a standard format."
            )

        # 3. Clean and format using pandas
        cleaned_txns = clean_and_format_transactions(raw_txns, date_format=date_format)

        return {
            "success": True,
            "filename": file.filename,
            "pages": page_count,
            "transactions": cleaned_txns,
            "transactions_count": len(cleaned_txns),
            "user_id": user["id"] if user else None
        }

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/convert/batch")
async def convert_batch(
    files: List[UploadFile] = File(...),
    passwords: Optional[str] = Form(None), # JSON string of index to password mapping
    bank: Optional[str] = Form("auto"),
    date_format: Optional[str] = Form("DD/MM/YYYY"),
    categorize: Optional[bool] = Form(True),
    gst: Optional[bool] = Form(True),
    user: Optional[dict] = Depends(verify_user)
):
    import json
    pwd_map = {}
    if passwords:
        try:
            pwd_map = json.loads(passwords)
        except Exception:
            pass

    async def process_file(idx: int, file: UploadFile):
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"batch_upload_{idx}_{file.filename}")
        
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            pwd = pwd_map.get(str(idx)) or pwd_map.get(idx)
            
            try:
                text, page_count = extract_text_from_pdf(temp_file_path, password=pwd)
            except Exception as e:
                err_msg = str(e).lower()
                if "password" in err_msg or "decrypt" in err_msg:
                    return {"index": idx, "filename": file.filename, "success": False, "error": "password_required", "message": "Password required"}
                return {"index": idx, "filename": file.filename, "success": False, "error": str(e)}

            if not text or len(text.strip()) < 20:
                return {"index": idx, "filename": file.filename, "success": False, "error": "Could not extract text from this PDF."}

            if not user and page_count > 1:
                return {"index": idx, "filename": file.filename, "success": False, "error": "Anonymous limit 1 page"}
                
            if user and supabase:
                try:
                    profile_res = supabase.table("profiles").select("tier, premium_expiry_date, credits").eq("id", user["id"]).execute()
                    tier = profile_res.data[0].get("tier", "registered") if profile_res.data else "registered"
                    expiry_date_str = profile_res.data[0].get("premium_expiry_date") if profile_res.data else None
                    credits_limit = profile_res.data[0].get("credits", 500) if profile_res.data else 500
                    
                    if tier == "subscribed" and expiry_date_str:
                        from datetime import datetime, timezone
                        now_iso = datetime.now(timezone.utc).isoformat()
                        if expiry_date_str < now_iso:
                            tier = "registered"
                            
                    if tier == "subscribed":
                        from datetime import datetime, timezone
                        today = datetime.now(timezone.utc)
                        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
                        conv_res = supabase.table("conversions").select("pages").eq("user_id", user["id"]).gte("created_at", month_start).execute()
                        pages_this_month = sum(c["pages"] for c in conv_res.data) if conv_res.data else 0
                        if pages_this_month + page_count > credits_limit:
                            return {"index": idx, "filename": file.filename, "success": False, "error": f"Monthly limit exceeded. Pages left: {max(0, credits_limit - pages_this_month)}"}
                    else:
                        from datetime import datetime, timezone
                        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                        conv_res = supabase.table("conversions").select("pages").eq("user_id", user["id"]).gte("created_at", today_start).execute()
                        pages_today = sum(c["pages"] for c in conv_res.data) if conv_res.data else 0
                        if pages_today + page_count > 5:
                            return {"index": idx, "filename": file.filename, "success": False, "error": f"Daily limit exceeded. Pages left today: {max(0, 5 - pages_today)}"}
                except Exception as e:
                    print(f"Failed to check batch quota: {e}")

            raw_txns = []
            try:
                raw_txns = parse_with_gemini(text[:600000], categorize=categorize, gst=gst)
            except Exception as e:
                print(f"Gemini AI failed for {file.filename}: {e}")

            if not raw_txns:
                try:
                    raw_txns = parse_pdf_natively(temp_file_path, password=pwd)
                except Exception as e:
                    return {"index": idx, "filename": file.filename, "success": False, "error": f"All parsers failed: {e}"}

            if not raw_txns:
                return {"index": idx, "filename": file.filename, "success": False, "error": "No transactions detected"}

            cleaned_txns = clean_and_format_transactions(raw_txns, date_format=date_format)

            return {
                "index": idx,
                "filename": file.filename,
                "success": True,
                "pages": page_count,
                "transactions": cleaned_txns,
                "transactions_count": len(cleaned_txns)
            }
        except Exception as e:
            return {"index": idx, "filename": file.filename, "success": False, "error": str(e)}
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    tasks = [process_file(idx, f) for idx, f in enumerate(files)]
    results = await asyncio.gather(*tasks)
    return {"results": results, "user_id": user["id"] if user else None}

@app.post("/api/download")
async def download_file(req: DownloadRequest):
    if not req.transactions:
        raise HTTPException(status_code=400, detail="Transactions list is empty.")
        
    temp_dir = tempfile.gettempdir()
    txns_data = [t.model_dump() for t in req.transactions]
    
    # Clean filename
    safe_filename = "".join([c for c in req.filename if c.isalpha() or c.isdigit() or c in ' ._-']).rstrip()
    safe_filename = os.path.splitext(safe_filename)[0] # remove extension if provided
    
    if req.format == "xlsx":
        file_path = os.path.join(temp_dir, f"{safe_filename}.xlsx")
        generate_excel_file(txns_data, file_path)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename_out = f"{safe_filename}.xlsx"
    elif req.format == "csv":
        file_path = os.path.join(temp_dir, f"{safe_filename}.csv")
        generate_csv_file(txns_data, file_path)
        media_type = "text/csv"
        filename_out = f"{safe_filename}.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid format specified. Must be 'xlsx' or 'csv'.")
        
    # Return as file response, deleted after send
    class DeleteOnCloseFileResponse(FileResponse):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def close(self) -> None:
            super().close()
            try:
                if os.path.exists(self.path):
                    os.remove(self.path)
            except Exception as e:
                print(f"Error removing temp file {self.path}: {e}")
                
    return DeleteOnCloseFileResponse(
        path=file_path, 
        media_type=media_type, 
        filename=filename_out,
        headers={"Access-Control-Expose-Headers": "Content-Disposition"}
    )
