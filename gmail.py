# agent.py — Render‑ready AI Invoice Agent (Single‑file FastAPI + English Dashboard)
# --------------------------------------------------------------------------------
# ========= Standard Library =========
import base64
import hashlib
import hmac
import imaplib
import io
import json
import mimetypes
import os
import pathlib
import re
import secrets
import smtplib
import ssl
import zipfile
from datetime import datetime

from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
from email import policy

# ========= Typing =========
from typing import List, Optional

# ========= Third-Party =========
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, create_engine, Session, select

# ----------------- App -----------------
app = FastAPI(title="AI Invoice Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
# ----------------- LOGGINS -----------------
import logging, time

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
)
logger = logging.getLogger("agent")


# --- GOOGLE OAUTH: BLOQUE COMPLETO (pegable) ---

import os
import json
import logging
import httpx

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# Logger (Render/uvicorn)
log = logging.getLogger("uvicorn.error")

# ========= 1) ENV saneadas =========
BASE_URL = (os.getenv("BASE_URL", "http://localhost:8000") or "").strip().rstrip("/")
APP_SECRET_KEY = (os.getenv("APP_SECRET_KEY") or "dev-secret").strip()

# OAuth (login con Google)
GOOGLE_CLIENT_ID = (os.getenv("GOOGLE_CLIENT_ID") or "").strip()
GOOGLE_CLIENT_SECRET = (os.getenv("GOOGLE_CLIENT_SECRET") or "").strip()

# Google Drive (Service Account) - solo para debug de existencia, NO imprimas el JSON
GOOGLE_CREDS_JSON = (os.getenv("GOOGLE_CREATDS_JSON") or os.getenv("GOOGLE_CREDS_JSON") or os.getenv("DRIVE_CREDS_JSON") or "credenciales.json").strip()
DRIVE_ROOT_FOLDER_ID = (os.getenv("DRIVE_ROOT_FOLDER_ID") or "1PoK57Hli-zG7ed8dnlKSEeDjCP8KAbuL").strip()

_creds_is_path = os.path.exists(GOOGLE_CREDS_JSON)
_creds_info = None
if not _creds_is_path:
    try:
        obj = json.loads(GOOGLE_CREDS_JSON)
        _creds_info = {
            "project_id": obj.get("project_id"),
            "client_email": obj.get("client_email"),
        }
    except Exception:
        _creds_info = {"note": "not a path; not plain JSON (maybe base64 or empty)"}

log.info(
    "ENV -> base_url=%s app_secret_set=%s oauth_id_len=%s has_oauth_secret=%s drive_root=%s drive_creds_path_exists=%s drive_creds_info=%s",
    BASE_URL,
    bool(APP_SECRET_KEY),
    len(GOOGLE_CLIENT_ID),
    bool(GOOGLE_CLIENT_SECRET),
    DRIVE_ROOT_FOLDER_ID,
    _creds_is_path,
    _creds_info,
)

# ========= 2) Crear app y SessionMiddleware =========
app = globals().get("app") or FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=APP_SECRET_KEY,
    same_site="lax",
    https_only=BASE_URL.startswith("https://"),
    max_age=60 * 60 * 24 * 7,
)

# ========= 3) Config OAuth =========
CLIENT_CONFIG = {
    "web": {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [f"{BASE_URL}/auth/callback"],
    }
}

# Scopes unificados
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def build_flow() -> Flow:
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise RuntimeError("Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET")
    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=OAUTH_SCOPES)
    flow.redirect_uri = f"{BASE_URL}/auth/callback"
    return flow

# ========= 4) Rutas OAuth =========
@app.get("/oauth/login")
async def oauth_login(request: Request):
    flow = build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    request.session["state"] = state
    log.info("OAuth: generated auth_url (client_id suffix=%s)", GOOGLE_CLIENT_ID[-20:] if GOOGLE_CLIENT_ID else None)
    return RedirectResponse(auth_url)

@app.get("/auth/callback")
async def oauth_callback(request: Request):
    flow = build_flow()
    flow.redirect_uri = f"{BASE_URL}/auth/callback"

    code = request.query_params.get("code")
    if not code:
        raise HTTPException(400, "Missing authorization code")

    state = request.query_params.get("state")
    saved_state = request.session.get("state")
    if saved_state and state and state != saved_state:
        raise HTTPException(400, "State mismatch")
    request.session.pop("state", None)

    try:
        flow.fetch_token(code=code, include_client_id=True)
    except Exception as e:
        log.exception("Token exchange failed. redirect_uri=%s request.url=%s", flow.redirect_uri, str(request.url))
        raise HTTPException(400, f"Token exchange failed: {e}")

    creds = flow.credentials

    # Preferido: Gmail profile
    email = None
    try:
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile = svc.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress")
    except Exception as e:
        log.warning("gmail profile fetch failed: %s", e)

    # Fallback: userinfo
    userinfo = {}
    if not email:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {creds.token}"},
                )
                if r.status_code == 200:
                    userinfo = r.json()
                    email = userinfo.get("email") or email
        except Exception as e:
            log.warning("userinfo fetch failed: %s", e)

    if not email:
        raise HTTPException(500, "Could not determine account email. Enable Gmail API for your project and retry.")

    # Guardar credenciales (si existe función global save_creds)
    if "save_creds" in globals() and callable(globals()["save_creds"]):
        try:
            globals()["save_creds"](email, creds)
        except Exception as e:
            log.warning("save_creds failed: %s", e)

    # Persistir sesión
    request.session["user"] = {
        "email": email,
        "name": userinfo.get("name"),
        "picture": userinfo.get("picture"),
        "id": userinfo.get("id"),
    }
    request.session["creds"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if isinstance(creds.scopes, set) else creds.scopes,
    }

    # Setear sessionStorage.ai_email y redirigir a /app
    email_js = json.dumps(email)
    return HTMLResponse(
        f'<script>'
        f'  try {{ sessionStorage.setItem("ai_email", {email_js}); }} catch (e) {{}}'
        f'  location.replace("/app");'
        f'</script>'
    )

# ========= 5) Diagnóstico =========
@app.get("/oauth/debug")
async def oauth_debug():
    return JSONResponse({
        "client_id_len": len(GOOGLE_CLIENT_ID),
        "client_id_suffix": GOOGLE_CLIENT_ID[-20:] if GOOGLE_CLIENT_ID else None,
        "has_secret": bool(GOOGLE_CLIENT_SECRET),
        "redirect_uri": f"{BASE_URL}/auth/callback",
        "drive_root": DRIVE_ROOT_FOLDER_ID,
        "drive_creds_is_path": _creds_is_path,
        "drive_creds_info": _creds_info,
    })



@app.get("/login")
async def login_get_router(request: Request):
    # ¿La UI está llamando con fetch/XHR?
    def wants_json() -> bool:
        accept = (request.headers.get("accept") or "").lower()
        xrw    = (request.headers.get("x-requested-with") or "").lower()
        sfm    = (request.headers.get("sec-fetch-mode") or "").lower()
        return ("application/json" in accept) or (xrw == "xmlhttprequest") or (sfm in ("cors", "same-origin"))

    # 1) Ya hay sesión -> /app
    user = request.session.get("user") or {}
    if user.get("email"):
        if wants_json():
            return JSONResponse({"connected": True, "email": user["email"], "next": "/app"})
        return RedirectResponse("/app", status_code=303)

    # 2) ?email= y tokens en BD -> crear sesión y /app
    email = request.query_params.get("email")
    if email:
        try:
            _ = get_creds(email)  # lanza si no existen credenciales
            request.session["user"] = {"email": email}
            if wants_json():
                return JSONResponse({"connected": True, "email": email, "next": "/app"})
            return RedirectResponse("/app", status_code=303)
        except Exception:
            pass  # sin tokens -> seguir a OAuth

    # 3) Sin sesión -> iniciar OAuth
    if wants_json():
        return JSONResponse({"connected": False, "next": "/oauth/login"})
    return RedirectResponse("/oauth/login", status_code=303)


# --- FIN BLOQUE GOOGLE OAUTH ---









# ----------------- Config -----------------
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
ATTACH_DIR = DATA_DIR / "attachments"
ATTACH_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "agent.db"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

# Opcional: correo por defecto del contador
ACCOUNTANT_EMAIL = os.getenv("ACCOUNTANT_EMAIL")  # optional


engine = create_engine(f"sqlite:///{DB_PATH}")
# ----------------- GMAIL -----------------

def imap_connect(email_addr: str, app_password: str) -> imaplib.IMAP4_SSL:
    imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    # Si la clave es la App Password con espacios, Gmail la acepta tal cual.
    imap.login(email_addr, app_password)
    return imap

def _decode_header_value(val: str) -> str:
    try:
        return str(make_header(decode_header(val or "")))
    except Exception:
        return val or ""

# ---------- IMAP: probar login con correo + App Password ----------
class ImapLoginBody(BaseModel):
    email: str
    app_password: str  # App Password de 16 chars

@app.post("/imap/test")
async def imap_test(body: ImapLoginBody):
    try:
        imap = imap_connect(body.email, body.app_password)
        imap.logout()
        return {"ok": True}
    except imaplib.IMAP4.error as e:
        raise HTTPException(401, f"IMAP auth failed: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"IMAP error: {str(e)}")
# ---------- DESCARGAR FACTURAS Y SELECCIÓN MÚLTIPLE Y ENVÍO DE CORREOS ----------

# ---- Forward selected emails (robusto con select + fallback) ----
class ForwardBody(BaseModel):
    email: EmailStr
    app_password: str
    to: EmailStr
    uids: list[str]

@app.post("/imap/forward")
async def imap_forward(body: ForwardBody):
    if not body.uids:
        raise HTTPException(400, "uids required")

    # 1) IMAP: conectar
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(body.email, body.app_password)
    except imaplib.IMAP4.error as e:
        raise HTTPException(401, f"IMAP auth failed: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"IMAP connect error: {str(e)}")

    # 2) Seleccionar buzón (All Mail -> fallback INBOX) y validar
    try:
        typ, _ = imap.select('"[Gmail]/All Mail"', readonly=True)
        if typ != 'OK':
            typ, _ = imap.select("INBOX", readonly=True)
        if typ != 'OK':
            try: imap.logout()
            except: pass
            raise HTTPException(500, "IMAP select failed (All Mail and INBOX)")
    except Exception as e:
        try: imap.logout()
        except: pass
        raise HTTPException(500, f"IMAP select error: {str(e)}")

    # 3) Descargar los RFC822 de cada UID
    originals = []
    try:
        for uid in body.uids:
            uid_b = uid.encode() if isinstance(uid, str) else uid
            typ, msg_data = imap.uid('FETCH', uid_b, '(RFC822)')
            if typ == 'OK' and msg_data and isinstance(msg_data[0], tuple):
                originals.append(msg_data[0][1])
    except Exception as e:
        try: imap.logout()
        except: pass
        raise HTTPException(500, f"IMAP fetch error: {str(e)}")
    finally:
        try: imap.logout()
        except: pass

    if not originals:
        return {"ok": False, "sent": 0, "detail": "No messages fetched"}

    # 4) SMTP: reenviar como .eml adjunto
    try:
        smtp = smtplib.SMTP("smtp.gmail.com", 587)
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(body.email, body.app_password)
    except smtplib.SMTPException as e:
        raise HTTPException(401, f"SMTP auth failed: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"SMTP connect error: {str(e)}")

    sent = 0
    try:
        for raw in originals:
            orig = BytesParser(policy=policy.default).parsebytes(raw)
            fwd = EmailMessage()
            fwd["From"] = body.email
            fwd["To"] = body.to
            subj = orig.get("Subject", "") or ""
            fwd["Subject"] = f"Fwd: {subj}"
            fwd.set_content("Forwarded message attached.\n")
            fwd.add_attachment(raw, maintype="message", subtype="rfc822", filename="original.eml")
            smtp.send_message(fwd)
            sent += 1
    finally:
        try: smtp.quit()
        except: pass

    return {"ok": True, "sent": sent}


# ---- Download selected as a ZIP (.eml each) (robusto con select + fallback) ----
class ZipBody(BaseModel):
    email: EmailStr
    app_password: str
    uids: list[str]

@app.post("/imap/download_zip")
async def imap_download_zip(body: ZipBody):
    if not body.uids:
        raise HTTPException(400, "uids required")

    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(body.email, body.app_password)
    except imaplib.IMAP4.error as e:
        raise HTTPException(401, f"IMAP auth failed: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"IMAP connect error: {str(e)}")

    # Selección de buzón con fallback
    try:
        typ, _ = imap.select('"[Gmail]/All Mail"', readonly=True)
        if typ != 'OK':
            typ, _ = imap.select("INBOX", readonly=True)
        if typ != 'OK':
            try: imap.logout()
            except: pass
            raise HTTPException(500, "IMAP select failed (All Mail and INBOX)")
    except Exception as e:
        try: imap.logout()
        except: pass
        raise HTTPException(500, f"IMAP select error: {str(e)}")

    mem = io.BytesIO()
    try:
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for uid in body.uids:
                uid_b = uid.encode() if isinstance(uid, str) else uid
                typ, data = imap.uid('FETCH', uid_b, '(RFC822)')
                if typ == 'OK' and data and isinstance(data[0], tuple):
                    zf.writestr(f"{uid}.eml", data[0][1])
    except Exception as e:
        try: imap.logout()
        except: pass
        raise HTTPException(500, f"IMAP fetch/zip error: {str(e)}")
    finally:
        try: imap.logout()
        except: pass

    mem.seek(0)
    return StreamingResponse(
        mem,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="emails.zip"'}
    )



# ---------- IMAP: búsqueda por palabra clave (literal por defecto) ----------


class ImapSearchBody(BaseModel):
    email: EmailStr
    app_password: str
    query: str                  # ← requerido; lo envía el usuario
    max_results: int = 50

def _decode_header_value(h: str) -> str:
    parts = decode_header(h or '')
    out = []
    for txt, enc in parts:
        out.append(txt.decode(enc or 'utf-8', errors='replace') if isinstance(txt, bytes) else str(txt))
    return ''.join(out)

def _quote_imap(s: str) -> str:
    # Escapa comillas internas para que el argumento quoted no rompa el comando IMAP
    return (s or '').replace('"', r'\"')

# Detecta si el usuario ya usa operadores de Gmail (entonces no tocamos la query)
OPS_RE = re.compile(r'(^|\s)(in:|from:|to:|subject:|newer_than|older_than|filename:|has:|before:|after:|OR|AND)\b', re.I)

def _to_gmail_raw(q: str) -> str:
    """
    - Si NO hay operadores de Gmail, buscamos en todo el buzón y forzamos literal por palabra:
        "amazon invoice" -> in:anywhere "amazon" "invoice"
    - Si hay operadores, respetamos tal cual.
    """
    q = (q or '').strip()
    if not q:
        return q
    if OPS_RE.search(q):
        return q
    terms = [t for t in q.split() if t]
    quoted = ' '.join(f'"{_quote_imap(t)}"' for t in terms)
    return f'in:anywhere {quoted}'

@app.post("/imap/search")
async def imap_search(body: ImapSearchBody):
    if not body.query or not body.query.strip():
        raise HTTPException(400, "query is required")

    # Conexión
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(body.email, body.app_password)
    except imaplib.IMAP4.error as e:
        raise HTTPException(401, f"IMAP auth failed: {str(e)}")
    except Exception as e:
        # Otro error de red/ssl
        raise HTTPException(500, f"IMAP connect error: {str(e)}")

    try:
        # Buscar en All Mail (todo Gmail); si falla, INBOX
        typ, _ = imap.select('"[Gmail]/All Mail"', readonly=True)
        if typ != 'OK':
            imap.select("INBOX", readonly=True)

        raw = _to_gmail_raw(body.query)
        # Log para depurar qué exactamente mandamos:
        try:
            log.info(f"X-GM-RAW query => {raw}")
        except Exception:
            pass

        # 🟢 IMPORTANTE: el argumento va entre comillas, con comillas internas escapadas
        typ, data = imap.uid('SEARCH', 'X-GM-RAW', f'"{_quote_imap(raw)}"')

        # Fallback: búsqueda básica por texto si X-GM-RAW no devolvió nada
        if typ != 'OK' or not data or not data[0]:
            fallback_terms = [t for t in (body.query.strip().split()) if t]
            fallback_literal = ' '.join(f'"{_quote_imap(t)}"' for t in fallback_terms) or '""'
            typ, data = imap.search(None, 'TEXT', fallback_literal)

        uids = data[0].split() if data and data[0] else []
        if not uids:
            return {"count": 0, "items": []}

        # Tomar más recientes
        if body.max_results:
            uids = uids[-body.max_results:]

        items = []
        for uid in reversed(uids):  # recientes primero
            # Asegurar tipo bytes para FETCH
            uid_bytes = uid if isinstance(uid, (bytes, bytearray)) else str(uid).encode()
            typ, msg_data = imap.uid('FETCH', uid_bytes, '(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])')
            if typ != 'OK' or not msg_data:
                continue
            raw_msg = b''.join(part[1] for part in msg_data if isinstance(part, tuple))
            msg = message_from_bytes(raw_msg) if raw_msg else None
            items.append({
                "uid": (uid_bytes.decode(errors='ignore') if isinstance(uid_bytes, (bytes, bytearray)) else str(uid)),
                "subject": _decode_header_value(msg.get('Subject', '') if msg else ''),
                "from": _decode_header_value(msg.get('From', '') if msg else ''),
                "date": (msg.get('Date', '') if msg else '')
            })

        return {"count": len(items), "items": items}

    except HTTPException:
        # Re-levanta HTTPException tal cual
        raise
    except Exception as e:
        # Log del error real antes de responder 500 para depurar
        try:
            log.exception("IMAP search error")
        except Exception:
            pass
        raise HTTPException(500, f"IMAP error: {str(e)}")
    finally:
        try:
            imap.logout()
        except Exception:
            pass

# ----------------- API: BUSCADO DESCARGAR Y REENVIOS DE CORREOS -----------------
class SearchBody(BaseModel):
    access_key: str
    email: str
    query: str = 'newer_than:12w (subject:(invoice OR factura) OR has:attachment filename:(pdf OR xml))'
    max_results: int = 50

@app.post("/api/search")
async def api_search(body: SearchBody):
    if not verify_access_key(body.access_key):
        raise HTTPException(403, "Invalid access key")

    creds = get_creds(body.email)
    svc = gmail_service(creds)
    results = svc.users().messages().list(
        userId="me", q=body.query, maxResults=min(body.max_results, 100)
    ).execute()
    messages = results.get("messages", [])
    out = []
    for m in messages:
        msg = svc.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"]
        ).execute()
        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        out.append({
            "id": m["id"],
            "threadId": msg.get("threadId"),
            "subject": headers.get("subject"),
            "from": headers.get("from"),
            "date": headers.get("date"),
        })
    return {"count": len(out), "items": out}


class DownloadBody(BaseModel):
    access_key: str
    email: str
    message_id: str

@app.post("/api/download")
async def api_download(body: DownloadBody):
    if not verify_access_key(body.access_key):
        raise HTTPException(403, "Invalid access key")

    creds = get_creds(body.email)
    svc = gmail_service(creds)

    msg = svc.users().messages().get(userId="me", id=body.message_id).execute()
    saved = []

    def walk(parts, acc):
        for p in parts or []:
            if p.get("filename") and p.get("body", {}).get("attachmentId"):
                acc.append(p)
            walk(p.get("parts", []) or [], acc)

    parts = []
    walk(msg.get("payload", {}).get("parts", []) or [], parts)

    for part in parts:
        mime = part.get("mimeType", "")
        filename = part.get("filename") or "attachment.bin"
        att_id = part.get("body", {}).get("attachmentId")
        if att_id and (filename.lower().endswith((".pdf", ".xml")) or mime in ["application/pdf", "text/xml", "application/xml"]):
            att = svc.users().messages().attachments().get(
                userId="me", messageId=body.message_id, id=att_id
            ).execute()
            data = base64.urlsafe_b64decode(att.get("data", "")) if att.get("data") else b""
            path = ATTACH_DIR / filename
            with open(path, "wb") as f:
                f.write(data)
            saved.append(str(path))
    return {"saved": saved}


class ForwardBody(BaseModel):
    access_key: str
    email: str
    message_id: str
    to: Optional[str] = None  # defaults to ACCOUNTANT_EMAIL

@app.post("/api/forward")
async def api_forward(body: ForwardBody):
    if not verify_access_key(body.access_key):
        raise HTTPException(403, "Invalid access key")

    to_addr = body.to or ACCOUNTANT_EMAIL
    if not to_addr:
        raise HTTPException(400, "Missing 'to' or ACCOUNTANT_EMAIL env var")

    creds = get_creds(body.email)
    svc = gmail_service(creds)

    full = svc.users().messages().get(
        userId="me", id=body.message_id, format="metadata", metadataHeaders=["Subject"]
    ).execute()
    subject = "FWD: " + next((h["value"] for h in full.get("payload", {}).get("headers", []) if h["name"] == "Subject"), "(no subject)")
    body_txt = "Forwarded by AI Invoice Agent.\n\n"

    mime = f"From: me\nTo: {to_addr}\nSubject: {subject}\n\n{body_txt}"
    raw = base64.urlsafe_b64encode(mime.encode("utf-8")).decode("utf-8")
    sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"status": "ok", "id": sent.get("id")}


class BulkForwardBody(BaseModel):
    access_key: str
    email: str
    query: str = 'newer_than:12w (subject:(invoice OR factura) OR has:attachment filename:(pdf OR xml))'
    to: Optional[str] = None
    max_results: int = 100

@app.post("/api/bulk-forward")
async def api_bulk_forward(body: BulkForwardBody):
    if not verify_access_key(body.access_key):
        raise HTTPException(403, "Invalid access key")

    to_addr = body.to or ACCOUNTANT_EMAIL
    if not to_addr:
        raise HTTPException(400, "Missing 'to' or ACCOUNTANT_EMAIL env var")

    creds = get_creds(body.email)
    svc = gmail_service(creds)
    results = svc.users().messages().list(
        userId="me", q=body.query, maxResults=min(body.max_results, 100)
    ).execute()
    messages = results.get("messages", [])
    sent_ids: List[str] = []
    for m in messages:
        msg_id = m["id"]
        full = svc.users().messages().get(
            userId="me", id=msg_id, format="metadata", metadataHeaders=["Subject"]
        ).execute()
        subject = "FWD: " + next((h["value"] for h in full.get("payload", {}).get("headers", []) if h["name"] == "Subject"), "(no subject)")
        body_txt = "Forwarded by AI Invoice Agent.\n\n"
        mime = f"From: me\nTo: {to_addr}\nSubject: {subject}\n\n{body_txt}"
        raw = base64.urlsafe_b64encode(mime.encode("utf-8")).decode("utf-8")
        sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
        sent_ids.append(sent.get("id"))
    return {"forwarded": len(sent_ids), "ids": sent_ids}


# ========= IMAP → Save image attachments to Google Drive =========
from typing import List, Optional
from pydantic import BaseModel, EmailStr
from fastapi import HTTPException

class ImapSaveToDriveImagesBody(BaseModel):
    email: EmailStr
    app_password: str
    uids: List[str]
    group_by_year_month: bool = True
    drive_root_folder_id: Optional[str] = None    # si no viene, usa ENV
    creds_json_path: Optional[str] = None         # si no viene, usa ENV

@app.post("/imap/save_to_drive_images")
async def imap_save_to_drive_images(body: ImapSaveToDriveImagesBody):
    import os, io, json, base64, imaplib, email, mimetypes
    from datetime import datetime
    from email import policy
    from email.parser import BytesParser
    from email.utils import parsedate_to_datetime
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload

    # ---------- 1) Login IMAP ----------
    def _open_imap(addr: str, app_pass: str):
        host = "imap.gmail.com"
        try:
            imap = imaplib.IMAP4_SSL(host)
            imap.login(addr, app_pass)
            rv, _ = imap.select("INBOX")
            if rv != "OK":
                for box in ('"[Gmail]/All Mail"', '"[Gmail]/Todos"', '"All Mail"'):
                    rv, _ = imap.select(box)
                    if rv == "OK":
                        break
            return imap
        except imaplib.IMAP4.error as e:
            raise HTTPException(400, f"IMAP auth failed: {e}")

    # ---------- 2) Drive client (ruta | JSON plano | base64) ----------
    def _build_drive(creds_source: Optional[str]):
        scopes = ["https://www.googleapis.com/auth/drive.file"]
        if creds_source and os.path.exists(creds_source):
            c = service_account.Credentials.from_service_account_file(creds_source, scopes=scopes)
            return build("drive", "v3", credentials=c)
        if creds_source:
            try:
                obj = json.loads(creds_source)
                c = service_account.Credentials.from_service_account_info(obj, scopes=scopes)
                return build("drive", "v3", credentials=c)
            except Exception:
                pass
        try:
            dec = base64.b64decode((creds_source or "").encode("utf-8")).decode("utf-8")
            obj = json.loads(dec)
            c = service_account.Credentials.from_service_account_info(obj, scopes=scopes)
            return build("drive", "v3", credentials=c)
        except Exception as e:
            raise HTTPException(500, f"Drive credentials error: {e}")

    # ---------- 3) Helpers Drive / Attachments ----------
    def _ensure_folder(drive, parent_id: str, name: str) -> str:
        safe = (name or "").replace("'", "\\'")
        q = f"'{parent_id}' in parents and name = '{safe}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = drive.files().list(q=q, fields="files(id)").execute()
        files = (res or {}).get("files") or []
        if files:
            return files[0]["id"]
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
        created = drive.files().create(body=meta, fields="id").execute()
        return created["id"]

    def _ensure_path(drive, root_id: str, year: str, month: str) -> str:
        p = _ensure_folder(drive, root_id, year)
        p = _ensure_folder(drive, p, month)
        return p

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif")
    def _is_image(filename: str, ctype: str) -> bool:
        fn = (filename or "").lower()
        if any(fn.endswith(ext) for ext in IMAGE_EXTS):
            return True
        return (ctype or "").startswith("image/")

    def _upload_bytes(drive, folder_id: str, filename: str, content: bytes, ctype: str):
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=(ctype or "application/octet-stream"), resumable=False)
        meta = {"name": filename, "parents": [folder_id]}
        created = drive.files().create(body=meta, media_body=media, fields="id,webViewLink,webContentLink").execute()
        return {"file_id": created["id"], "webViewLink": created.get("webViewLink"), "webContentLink": created.get("webContentLink")}

    def _safe_name(name: str) -> str:
        n = (name or "").strip() or "image"
        return n.replace("/", "_").replace("\\", "_").replace("\0", "")

    # ---------- 4) Preparar Drive ----------
    sa_source = body.creds_json_path or os.getenv("GOOGLE_CREDS_JSON") or os.getenv("DRIVE_CREDS_JSON") or "credenciales.json"
    drive = _build_drive(sa_source)
    root_folder_id = body.drive_root_folder_id or os.getenv("DRIVE_ROOT_FOLDER_ID") or "1PoK57Hli-zG7ed8dnlKSEeDjCP8KAbuL"

    # ---------- 5) IMAP: fetch por UID y subir imágenes ----------
    imap = _open_imap(body.email, body.app_password)

    results = []
    uploaded_total = 0

    try:
        for uid in body.uids:
            typ, data = imap.uid("fetch", str(uid), "(RFC822)")
            if typ != "OK" or not data:
                results.append({"uid": uid, "uploaded_count": 0, "files": [], "error": "fetch failed"})
                continue

            raw_bytes = None
            for part in data:
                if isinstance(part, tuple) and isinstance(part[1], (bytes, bytearray)):
                    raw_bytes = part[1]
                    break
            if not raw_bytes:
                results.append({"uid": uid, "uploaded_count": 0, "files": [], "error": "empty message"})
                continue

            msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

            try:
                dt = parsedate_to_datetime(msg.get("Date")) if msg.get("Date") else None
            except Exception:
                dt = None
            if not dt:
                dt = datetime.utcnow()
            year = f"{dt.year:04d}"
            month = f"{dt.month:02d}"

            uploaded = []
            for part in msg.walk():
                if part.is_multipart():
                    continue
                ctype = part.get_content_type() or ""
                fname = part.get_filename() or ""
                if not _is_image(fname, ctype):
                    continue
                content = part.get_payload(decode=True) or b""
                if not content:
                    continue

                dest_folder = _ensure_path(drive, root_folder_id, year, month) if body.group_by_year_month else root_folder_id

                if not fname.strip():
                    ext = mimetypes.guess_extension(ctype) or ".bin"
                    fname = f"image_uid{uid}{ext}"
                fname = _safe_name(fname)

                info = _upload_bytes(drive, dest_folder, fname, content, ctype)
                uploaded.append({"filename": fname, **info})

            uploaded_total += len(uploaded)
            results.append({"uid": uid, "uploaded_count": len(uploaded), "files": uploaded})
    finally:
        try: imap.logout()
        except Exception: pass

    return {"processed": len(body.uids), "uploaded_total": uploaded_total, "details": results}



# ----------------- DB Models -----------------
class UserToken(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str
    token_json: str  # serialized google credentials (includes refresh token)

class AppConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=1, primary_key=True)
    access_key_hash: Optional[str] = None
    salt: Optional[str] = None
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None

SQLModel.metadata.create_all(engine)

with Session(engine) as s:
    if not s.get(AppConfig, 1):
        s.add(AppConfig(id=1))
        s.commit()




# ----------------- Helpers (simple login: no tenants, no encryption) -----------------
def _flow() -> Flow:
    c = get_oauth_config()
    cfg = {
        "web": {
            "client_id": c["client_id"],
            "client_secret": c["client_secret"],
            "redirect_uris": [c["redirect_uri"]],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    return Flow.from_client_config(cfg, scopes=SCOPES, redirect_uri=c["redirect_uri"])


def gmail_service(creds: Credentials):
    return build("gmail", "v1", credentials=creds, cache_discovery=False)

def save_creds(email: str, creds: Credentials):
    with Session(engine) as s:
        row = s.exec(select(UserToken).where(UserToken.email == email)).first()
        if not row:
            row = UserToken(email=email, token_json=creds.to_json())
        else:
            row.token_json = creds.to_json()
        s.add(row)
        s.commit()

def get_creds(email: str) -> Credentials:
    with Session(engine) as s:
        row = s.exec(select(UserToken).where(UserToken.email == email)).first()
        if not row:
            raise HTTPException(401, "Not connected. Use /auth/google first")
        creds = Credentials.from_authorized_user_info(json.loads(row.token_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
            row.token_json = creds.to_json()
            s.add(row)
            s.commit()
        return creds
def _derive_key(raw_key: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", raw_key.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return base64.b64encode(dk).decode("utf-8")

def set_access_key_once(raw_key: str):
    with Session(engine) as s:
        cfg = s.get(AppConfig, 1)
        if cfg.access_key_hash:
            raise HTTPException(409, "Access key already set")
        salt = secrets.token_urlsafe(16)
        cfg.salt = salt
        cfg.access_key_hash = _derive_key(raw_key, salt)
        s.add(cfg); s.commit()

def verify_access_key(raw_key: str) -> bool:
    with Session(engine) as s:
        cfg = s.get(AppConfig, 1)
        if not (cfg and cfg.access_key_hash and cfg.salt):
            return False
        candidate = _derive_key(raw_key, cfg.salt)
        return hmac.compare_digest(candidate, cfg.access_key_hash)

def get_oauth_config() -> dict:
    with Session(engine) as s:
        cfg = s.get(AppConfig, 1)
        if not (cfg and cfg.google_client_id and cfg.google_client_secret and cfg.redirect_uri):
            raise HTTPException(503, "OAuth is not configured yet")
        return {
            "client_id": cfg.google_client_id,
            "client_secret": cfg.google_client_secret,
            "redirect_uri": cfg.redirect_uri,
        }
# --- First-time setup: cliente define su access_key y sus credenciales Google ---
class ConfigSetup(BaseModel):
    access_key: str
    google_client_id: str
    google_client_secret: str
    redirect_uri: str  # ej: https://<tu-servicio>.onrender.com/auth/google/callback

@app.post("/config/setup")
async def config_setup(body: ConfigSetup):
    set_access_key_once(body.access_key)
    with Session(engine) as s:
        cfg = s.get(AppConfig, 1)
        cfg.google_client_id = body.google_client_id
        cfg.google_client_secret = body.google_client_secret
        cfg.redirect_uri = body.redirect_uri
        s.add(cfg); s.commit()
    return {"ok": True}

@app.get("/config/status")
async def config_status():
    with Session(engine) as s:
        cfg = s.get(AppConfig, 1)
        ready = bool(cfg and cfg.access_key_hash and cfg.google_client_id and cfg.google_client_secret and cfg.redirect_uri)
        return {"ready": ready}


#login de inicio de sesion

# --- LOGIN DEMO: email + clave (App Password) vía IMAP ---
class LoginBody(BaseModel):
    email: str
    access_key: str  # aquí usamos este campo como App Password (16 caracteres)

@app.post("/login")
async def login(body: LoginBody):
    logger.info("IMAP login attempt email=%s", body.email)
    try:
        # intenta conectar por IMAP con correo + App Password
        imap = imap_connect(body.email, body.access_key)
        imap.logout()
        # éxito: mandamos al front a /imap
        return {"connected": True, "next": "/imap"}
    except imaplib.IMAP4.error as e:
        logger.warning("IMAP auth failed for %s: %s", body.email, str(e))
        raise HTTPException(401, f"IMAP auth failed: {str(e)}")
    except Exception as e:
        logger.exception("IMAP login exception for %s", body.email)
        raise HTTPException(500, f"IMAP error: {str(e)}")

# ----------------- Gate (simple: single ACCESS_KEY) -----------------
class Gate(BaseModel):
    access_key: str

@app.post("/gate")
async def gate(body: Gate):
    if not verify_access_key(body.access_key):
        raise HTTPException(403, "Invalid access key")
    return {"ok": True}

# ----------------- Google OAuth (simple, global CLIENT_ID/SECRET) -----------------
@app.get("/auth/google")
async def auth_google():
    flow = _flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return {"auth_url": auth_url}

@app.get("/auth/google/callback")
async def auth_callback(request: Request):
    flow = _flow()
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials

    svc = gmail_service(creds)
    email = svc.users().getProfile(userId="me").execute().get("emailAddress")
    save_creds(email, creds)

    # Volver a la app ya conectado
    return HTMLResponse(
        f'<script>'
        f'sessionStorage.setItem("ai_email","{email}");'
        f'window.location="/app?email={email}";'
        f'</script>'
    )






@app.get("/app", response_class=HTMLResponse)
async def app_page():
    return HTMLResponse(APP_HTML)

APP_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Invoice Agent — App</title>
<style>
  body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:#0b0d10;color:#e6eaf0;margin:0}
  main{max-width:800px;margin:6vh auto;padding:24px}
  .card{background:#0f1318;border:1px solid #1c2128;border-radius:16px;padding:20px}
  input,button{font-size:15px}
  button{padding:10px 14px;border-radius:12px;border:1px solid #2e7d32;background:#14ae5c;color:white;font-weight:700;cursor:pointer}
  input[type=text]{width:100%;padding:12px;border-radius:12px;border:1px solid #263040;background:#0b0d10;color:#e6eaf0;margin:8px 0}
  pre{white-space:pre-wrap}
</style>
</head>
<body>
<main class="card">

  <!-- View when NOT signed in (no Google button shown) -->
  <section id="login-view" style="display:none">
    <h2>Sign in required</h2>
    <p>Please sign in from the main dashboard to continue.</p>
  </section>

  <!-- View when signed in -->
  <section id="app-view" style="display:none">
    <h2>Connected</h2>
    <p id="who"></p>

    <div style="margin:12px 0">
      <input id="query" type="text" value="newer_than:4w (subject:(invoice OR bill) OR has:attachment filename:(pdf OR xml))">
      <button onclick="testSearch()">Test: Search 5</button>
    </div>

    <pre id="out"></pre>
  </section>

</main>

<script>
(async function init(){
  // 1) If the OAuth callback added ?email=..., store it and clean the URL
  const url = new URL(location.href);
  const qpEmail = url.searchParams.get('email');
  if (qpEmail) {
    try { sessionStorage.setItem('ai_email', qpEmail); } catch(e) {}
    url.searchParams.delete('email');
    try { history.replaceState(null, '', url.toString()); } catch(e) {}
  }

  // 2) Read email from sessionStorage
  let email = sessionStorage.getItem('ai_email');

  // 3) If no email yet, ask backend to sync session state
  if (!email) {
    try {
      const s = await fetch('/auth/status', { credentials: 'same-origin' }).then(r => r.json());
      if (s && s.connected && s.email) {
        sessionStorage.setItem('ai_email', s.email);
        email = s.email;
      }
    } catch (e) {
      console.warn('auth/status failed', e);
    }
  }

  // 4) Show proper view
  const loginView = document.getElementById('login-view');
  const appView   = document.getElementById('app-view');
  const who       = document.getElementById('who');

  if (email) {
    loginView.style.display = 'none';
    appView.style.display = '';
    who.textContent = 'Logged in as: ' + email;
  } else {
    loginView.style.display = '';
    appView.style.display = 'none';
  }
})();

async function testSearch(){
  const email = sessionStorage.getItem('ai_email');
  const access_key = sessionStorage.getItem('ai_access_key') || ''; // if you use /gate
  const query = document.getElementById('query').value;
  const res = await fetch('/api/search', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ access_key, email, query, max_results:5 })
  });
  const j = await res.json().catch(()=>({detail:'Error'}));
  document.getElementById('out').textContent = JSON.stringify(j,null,2);
}
</script>
</body>
</html>
"""
@app.get("/imap", response_class=HTMLResponse)
async def imap_ui():
    return HTMLResponse(IMAP_HTML)

IMAP_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IMAP — Connect & Search</title>
<style>
  body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:#0b0d10;color:#e6eaf0;margin:0}
  main{max-width:560px;margin:8vh auto;padding:24px}
  .card{background:#0f1318;border:1px solid #1c2128;border-radius:16px;padding:20px;box-shadow:0 6px 16px rgba(0,0,0,.25)}
  input,button{font-size:16px}
  .inp{width:100%;padding:12px;border-radius:12px;border:1px solid #263040;background:#0b0d10;color:#e6eaf0;margin:8px 0}
  .btn{padding:12px 14px;border-radius:12px;border:1px solid #2e7d32;background:#14ae5c;color:white;font-weight:700;cursor:pointer}
  .btn:disabled{opacity:.6;cursor:not-allowed}
  .row{display:flex;gap:8px;align-items:center}
  .searchbar{display:none;gap:8px;margin-top:12px}
  .ok{color:#7CFC8A}.err{color:#ff6b6b}
  .bulk{display:none;gap:8px;margin:12px 0;align-items:center;flex-wrap:wrap}
  ul.mail{padding-left:18px;margin:8px 0}
  ul.mail li{margin:8px 0}
</style>
</head>
<body>
<main>
  <div class="card">
    <h2>Connect to IMAP</h2>

    <input id="email" class="inp" type="text" placeholder="Email address">
    <input id="appPass" class="inp" type="password" placeholder="App Password (16 characters)">
    <button class="btn" id="connectBtn" onclick="connectIMAP()">Connect</button>
    <p id="msg"></p>

    <!-- Search bar (only visible after connection) -->
    <form id="searchForm" class="row searchbar" onsubmit="doSearch(event)">
      <input id="query" class="inp" type="text"
             placeholder="Search (e.g., subject:&quot;invoice&quot; newer_than:6m)">
      <button class="btn" type="submit" title="Search">🔎 Search</button>
    </form>

    <!-- Bulk actions (only visible when there is a selection) -->
    <div id="bulkBar" class="bulk">
      <label style="display:flex;gap:6px;align-items:center;">
        <input type="checkbox" id="selAll" onchange="toggleAll(this)"> Select all
      </label>
      <button class="btn" onclick="downloadSelected()">⬇️ Download</button>
      <button class="btn" onclick="forwardSelected()">📤 Forward</button>
      <button class="btn" id="btnSaveDrive" onclick="saveSelectedToDriveImages()">📤 Save to Drive</button>
      <span id="selCount" style="margin-left:auto;opacity:.85"></span>
    </div>

    <div id="out"><h3>Results (0)</h3></div>
  </div>
</main>

<script>
function say(t, ok){
  document.getElementById('msg').innerHTML =
    '<small class="'+(ok?'ok':'err')+'">'+t+'</small>';
}

/* Front no altera la query; el backend la interpreta tal cual */
function toGmailRaw(q){
  return (q || '').trim();
}

async function connectIMAP(){
  const email = document.getElementById('email').value.trim();
  const app_password = document.getElementById('appPass').value.trim();
  const btn = document.getElementById('connectBtn');
  if(!email || !app_password){ say('Please enter email and App Password', false); return; }

  btn.disabled = true; say('Connecting…', true);
  let res = await fetch('/imap/test', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ email, app_password })
  });
  let j={}; try{ j = await res.json(); }catch(_){}
  btn.disabled = false;

  if(res.ok && (j.ok || j.connected)){
    say('Connected ✔', true);
    sessionStorage.setItem('ai_email', email);
    sessionStorage.setItem('ai_app_password', app_password);
    document.getElementById('searchForm').style.display = 'flex';
    document.getElementById('query').focus();
    document.getElementById('out').innerHTML = '<h3>Results (0)</h3>';
  }else{
    say('Connection error: ' + (j.detail || 'failed'), false);
  }
}

let selected = new Set();
let lastItems = [];

function renderResults(items){
  lastItems = items || [];
  selected.clear();
  updateBulkBar();

  if (!items || !items.length){
    document.getElementById('out').innerHTML = '<h3>Results (0)</h3>';
    return;
  }

  const rows = items.map(x => `
    <li>
      <label style="display:flex;gap:8px;align-items:flex-start;">
        <input type="checkbox" class="rowchk" data-uid="${x.uid}" onchange="toggleSel(this)">
        <div>
          <code>${x.uid||''}</code> — <b>${(x.subject||'').replace(/</g,'&lt;')}</b><br>
          <small>${(x.from||'').replace(/</g,'&lt;')} — ${x.date||''}</small>
        </div>
      </label>
    </li>`).join('');
  document.getElementById('out').innerHTML =
    '<h3>Results ('+(items.length||0)+')</h3><ul class="mail">'+rows+'</ul>';
}

function toggleSel(chk){
  const uid = chk.dataset.uid;
  if (chk.checked) selected.add(uid); else selected.delete(uid);
  updateBulkBar();
}

function toggleAll(master){
  const chks = Array.from(document.querySelectorAll('.rowchk'));
  if (master.checked){
    chks.forEach(c => { c.checked = true; selected.add(c.dataset.uid); });
  } else {
    chks.forEach(c => { c.checked = false; });
    selected.clear();
  }
  updateBulkBar();
}

function updateBulkBar(){
  const bar = document.getElementById('bulkBar');
  const c = selected.size;
  bar.style.display = c>0 ? 'flex' : 'none';
  document.getElementById('selCount').textContent = c>0 ? `${c} selected` : '';
  if (c===0) document.getElementById('selAll').checked = false;

  // Habilita/Deshabilita Save to Drive según selección
  const btnSave = document.getElementById('btnSaveDrive');
  if (btnSave) btnSave.disabled = c === 0;
}

async function doSearch(e){
  e.preventDefault();
  const email = sessionStorage.getItem('ai_email') || document.getElementById('email').value.trim();
  const app_password = sessionStorage.getItem('ai_app_password') || document.getElementById('appPass').value.trim();
  const raw = document.getElementById('query').value.trim();
  if(!raw){ return; }

  document.getElementById('out').innerHTML = '<h3>Searching…</h3>';

  const res = await fetch('/imap/search', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ email, app_password, query: toGmailRaw(raw), max_results: 50 })
  });

  let j; try { j = await res.json(); } catch { j = {count:0, items:[], detail:'Invalid JSON'}; }
  if(!res.ok){
    document.getElementById('out').innerHTML =
      '<h3>Search error</h3><pre>'+ (j.detail || res.statusText) +'</pre>';
    return;
  }

  renderResults(j.items || []);
}

async function downloadSelected(){
  if (!selected.size) return;
  const email = sessionStorage.getItem('ai_email');
  const app_password = sessionStorage.getItem('ai_app_password');
  const res = await fetch('/imap/download_zip', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ email, app_password, uids:[...selected] })
  });
  if(!res.ok){ alert('Download failed'); return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'emails.zip'; a.click();
  URL.revokeObjectURL(url);
}

async function forwardSelected(){
  if (!selected.size) return;
  const to = prompt('Forward to (email):');
  if(!to) return;
  const email = sessionStorage.getItem('ai_email');
  const app_password = sessionStorage.getItem('ai_app_password');
  const res = await fetch('/imap/forward', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ email, app_password, to, uids:[...selected] })
  });
  const j = await res.json().catch(()=>({ok:false}));
  alert(j.ok ? `Forwarded ${j.sent} emails` : (j.detail || 'Forward failed'));
}

/* ===== Save to Drive: solo imágenes adjuntas ===== */
function setDriveBusy(busy){
  const b = document.getElementById('btnSaveDrive');
  if(!b) return;
  b.disabled = !!busy;
  b.textContent = busy ? 'Saving…' : '📤 Save to Drive';
  b.style.opacity = busy ? 0.6 : 1;
}

async function saveSelectedToDriveImages(){
  if(!selected.size){ alert('Selecciona al menos 1 correo'); return; }
  const email = sessionStorage.getItem('ai_email');
  const app_password = sessionStorage.getItem('ai_app_password');
  if(!email || !app_password){ alert('Conéctate primero con IMAP'); return; }

  setDriveBusy(true);
  try{
    const res = await fetch('/imap/save_to_drive_images', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        email,
        app_password,
        uids: [...selected],
        group_by_year_month: true
        // drive_root_folder_id y creds_json_path se toman de ENV en backend
      })
    });
    const j = await res.json().catch(()=>({}));
    if(!res.ok) throw new Error(j.detail || res.statusText || 'Request failed');
    const total = j?.uploaded_total ?? 0;
    alert(`Listo: ${total} imagen(es) guardadas en Drive.`);
  }catch(err){
    console.error(err);
    alert('Error al guardar en Drive: ' + (err.message || err));
  }finally{
    setDriveBusy(false);
  }
}
</script>
</body>
</html>
"""



@app.get("/", response_class=HTMLResponse)
async def ui_root():
    return HTMLResponse(DASHBOARD_HTML)

DASHBOARD_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Invoice Agent — Login</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:#0b0d10;color:#e6eaf0;margin:0}
    main{max-width:420px;margin:10vh auto;padding:24px}
    .card{background:#0f1318;border:1px solid #1c2128;border-radius:16px;padding:20px;box-shadow:0 6px 16px rgba(0,0,0,.25)}
    h1{font-size:20px;margin:0 0 16px 0}
    input,button{font-size:16px}
    input[type=text],input[type=password]{width:100%;padding:12px;border-radius:12px;border:1px solid #263040;background:#0b0d10;color:#e6eaf0;margin:8px 0}
    button{width:100%;padding:12px;border-radius:12px;border:1px solid #2e7d32;background:#14ae5c;color:white;font-weight:700;cursor:pointer;margin-top:10px}
    small{opacity:.85}
    .ok{color:#7CFC8A}
    .err{color:#ff6b6b}
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>Sign in with IMAP</h1>

      <input id="email" type="text" placeholder="Email address">
      <input id="accessKey" type="password" placeholder="App Password (16 characters)">
      <button id="loginBtn" onclick="login()">Sign in</button>
      <p id="msg"><small>Use your <b>Gmail App Password</b> (not your regular password). You must have 2FA and IMAP enabled.</small></p>
    </div>
  </main>

<script>
async function login(){
  const email = document.getElementById('email').value.trim();
  const access_key = document.getElementById('accessKey').value.trim();
  if(!email || !access_key){ show('Enter email and App Password', true); return; }

  sessionStorage.setItem('ai_email', email);
  sessionStorage.setItem('ai_app_password', access_key);

  const btn = document.getElementById('loginBtn');
  btn.disabled = true;

  let res;
  try{
    res = await fetch('/login', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ email, access_key })
    });
  }catch(_){
    btn.disabled = false;
    show('Network error', true);
    return;
  }

  btn.disabled = false;
  let j = {};
  try{ j = await res.json(); }catch(_){}

  if(res.ok && j.connected){
    window.location = j.next || '/imap';
  }else{
    show(j.detail || 'Invalid credentials', true);
  }
}

function show(msg, isErr){
  const el = document.getElementById('msg');
  el.innerHTML = '<small class="' + (isErr?'err':'ok') + '">' + msg + '</small>';
}
</script>
</body>
</html>
"""
