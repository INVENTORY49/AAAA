# agente.py — RAVIE: asistente conversacional + generación de arte (corregido y limpio)

# 📦 Core
import os, io, uuid, pathlib, json, base64, time, logging, mimetypes, re
from typing import Optional, List, Dict, Any, Literal

# 🌐 FastAPI (solo APIRouter aquí)
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 🖼️ Imágenes
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from PIL import ImageChops, ImageFilter  
# 🤖 OpenAI (SDK nuevo)
from openai import OpenAI, AuthenticationError
# (si usas APIError/PermissionDeniedError en algún try/except, impórtalos aquí también)

# Logger del agente
log = logging.getLogger("ravie-agent")

# Instancia del router (usado por @router.post)
router = APIRouter()
app = router

# logger (usa el mismo canal que el resto de tu app)
log = logging.getLogger("uvicorn.error")
def _ok(reply: str, state: dict, preview_url: Optional[str] = None) -> JSONResponse:
    reply = reply or ""
    if not isinstance(preview_url, str):
        preview_url = None
    return JSONResponse(content={
        "ok": True, "reply": reply, "state": state, "preview_url": preview_url
    })

def _fail(msg: str, state: dict) -> JSONResponse:
    return JSONResponse(
        content={"ok": False, "error": msg, "state": state, "preview_url": None},
        status_code=200
    )
def make_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY")

    org = os.getenv("OPENAI_ORG_ID", "").strip()
    kwargs = {"api_key": api_key}

    if org and (org.startswith("org_") or org.startswith("org-")):
        kwargs["organization"] = org
        logging.getLogger("ravie-agent").info("[openai] usando header organization=%s", org)
    else:
        logging.getLogger("ravie-agent").info("[openai] sin header organization (se usa la org por defecto del API key)")

    return OpenAI(**kwargs)


def make_openai_client_no_org() -> OpenAI:
    """
    Cliente 'limpio' que NO envía OpenAI-Organization.
    Quita temporalmente OPENAI_ORG_ID / OPENAI_ORGANIZATION del entorno.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY")

    backup = {k: os.environ.get(k) for k in ("OPENAI_ORG_ID", "OPENAI_ORGANIZATION")}
    for k in ("OPENAI_ORG_ID", "OPENAI_ORGANIZATION"):
        os.environ.pop(k, None)

    try:
        client = OpenAI(api_key=api_key)  # cliente sin org
        logging.getLogger("ravie-agent").info("[openai] cliente SIN organization (fallback)")
        return client
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

FAQ_PATTERNS = [
    (re.compile(r"\biphone\s*11\b", re.I),
     "¡Cero problema! Te llega para tu iPhone 11 exacto. Lo que ves en pantalla es un mockup de ejemplo. ¿Quieres que te lo prepare con tu modelo?"),
    (re.compile(r"\bprecio|vale|cu[áa]nto cuesta\b", re.I),
     "Desde $59.900 COP (según modelo y acabado). ¿Cuál es tu modelo y qué estilo te gustaría?"),
    (re.compile(r"\benv[ií]o|domicilio|cu[aá]nto tarda\b", re.I),
     "Enviamos a todo el país, 2–5 días hábiles según ciudad. ¿Para qué ciudad sería?"),
    (re.compile(r"\bmaterial|resistente|antigolpe", re.I),
     "Case resistente anti-golpes, bordes flexibles, opción mate o brillante. ¿Cuál prefieres?")
]

def quick_faq_reply(text: str) -> str | None:
    t = (text or "").strip()
    for rx, ans in FAQ_PATTERNS:
        if rx.search(t):
            return ans
    # compatibilidad genérica
    if re.search(r"\b(iPhone|Samsung|Xiaomi|Motorola)\b", t, re.I) and re.search(r"\bproblema|sirve|funciona\b", t, re.I):
        return "Sí, trabajamos ese modelo sin problema. Lo que ves es solo un ejemplo/maqueta. ¿Quieres que lo armemos para tu modelo?"
    return None


CONFIRM_YES_RX = re.compile(r"\b(s[ií]|dale|hazlo|mu[eé]str(a|ame)|ok|listo|de una|vamos)\b", re.I)
CONFIRM_NO_RX  = re.compile(r"\b(no|espera|todav[ií]a|a[úu]n no|despu[eé]s)\b", re.I)

def wants_preview_yes(t: str) -> bool: return bool(CONFIRM_YES_RX.search(t or ""))
def wants_preview_no(t: str)  -> bool: return bool(CONFIRM_NO_RX.search(t or ""))

# ====== Fases (sin pedir modelo) ======
FASES = [
    "inicio",
    "pidiendo_tema",
    "checkpoint_preview",   # ⬅️ NUEVA
    "pidiendo_estilo",      # (opcional: si el user pide cambiar estilo/colores/texto)
    "listo_para_generar",
    "generando_preview",
    "ajustando",
    "checkout",
]

def _init_state(s: Dict[str, Any] | None) -> Dict[str, Any]:
    s = (s or {}).copy()
    s.setdefault("fase", "pidiendo_tema")
    s.setdefault("tema", None)
    s.setdefault("protagonistas", None)
    s.setdefault("estilo", None)
    s.setdefault("colores", None)
    s.setdefault("texto", None)
    s.setdefault("detalles_extra", [])
    s.setdefault("ultimo_preview", None)
    s.setdefault("final", False)

    # 👇 NUEVO: candados para no generar sin permiso
    s.setdefault("asked_preview", False)    # ya pregunté “¿Te muestro un preview?”
    s.setdefault("confirm_preview", False)  # usuario dijo SÍ
    return s

def _set_fase(s: Dict[str, Any], fase: str):
    s["fase"] = fase

def _brief_minimo(s: Dict[str, Any]) -> bool:
    # Generamos cuando hay al menos tema/protagonistas + estilo
    return bool((s.get("tema") or s.get("protagonistas")) and s.get("estilo"))



# ------------------ Logging ------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")

# ------------------ Config ------------------
BASE_DIR     = pathlib.Path("/mnt/data")
ASSETS_DIR   = BASE_DIR / "assets"
PREVIEWS_DIR = BASE_DIR / "previews"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# Drive (acepta ambas variables)
GOOGLE_CREDS_VALUE = os.getenv("GOOGLE_CREDS_JSON") or os.getenv("GOOGLE_CREED_JSON", "")
DRIVE_ROOT_ID      = os.getenv("DRIVE_ROOT_ID", "")
SUBFOLDER_NAME     = os.getenv("SUBFOLDER_NAME", "Agente")
BASE_NAME          = os.getenv("BASE_NAME", "base.png")
CAMARAS_NAME       = os.getenv("CAMARAS_NAME", "camaras.png")
OPENAI_ORG_ID      = os.getenv("OPENAI_ORG_ID")

# ------------------ CAMARA SUPERPUESTA ------------------
def ensure_camera_overlay_from_drive() -> pathlib.Path:
    """
    Descarga (si hace falta) el overlay de cámaras desde Drive y lo deja en assets.
    Usa la variable CAMARAS_NAME definida en el entorno.
    """
    local = ASSETS_DIR / "camera_overlay.png"
    if local.exists():
        return local
    if not (DRIVE_ROOT_ID and GOOGLE_CREDS_VALUE):
        raise RuntimeError("No hay camera_overlay.png local y Drive no está configurado")

    sub = drive_find_subfolder(DRIVE_ROOT_ID, SUBFOLDER_NAME)
    if not sub:
        raise RuntimeError(f"No se encontró subcarpeta '{SUBFOLDER_NAME}' en Drive")

    # 👉 Aquí usamos CAMARAS_NAME
    f = drive_find_file(sub["id"], CAMARAS_NAME)
    if not f:
        raise RuntimeError(f"No se encontró '{CAMARAS_NAME}' en Drive")

    local.write_bytes(drive_download(f["id"]))
    return local
def apply_camera_overlay(canvas_rgba: Image.Image,
                         overlay_rgba: Image.Image,
                         cam_bbox: tuple[int,int,int,int],
                         scale: float = 0.96,
                         offset: tuple[int,int] = (0, 0)) -> Image.Image:
    """
    Pega el overlay de cámaras (PNG con alpha) encima del resultado final.
    - cam_bbox: (left, top, right, bottom) del área de la 'isla' en base.png
    - scale: factor para hacerlo un poquito más pequeño que la ventana
    - offset: ajustes finos (dx, dy) en píxeles
    """
    L, T, R, B = cam_bbox
    bw, bh = max(1, R - L), max(1, B - T)
    tgt_w = int(round(bw * scale))
    tgt_h = int(round(bh * scale))

    ov = overlay_rgba
    # mantenemos proporciones del overlay
    ar_overlay = ov.width / max(1, ov.height)
    ar_target  = tgt_w / max(1, tgt_h)
    if ar_overlay > ar_target:
        # limita por ancho
        new_w = tgt_w
        new_h = int(round(new_w / ar_overlay))
    else:
        # limita por alto
        new_h = tgt_h
        new_w = int(round(new_h * ar_overlay))
    ov = ov.resize((max(1,new_w), max(1,new_h)), Image.LANCZOS)

    # centramos en cam_bbox y aplicamos offset
    dx, dy = offset
    x = L + (bw - ov.width)//2 + dx
    y = T + (bh - ov.height)//2 + dy

    out = canvas_rgba.copy()
    out.alpha_composite(ov, dest=(x, y))
    return out


# ------------------ RECORTE DE CAMARA INTELIGENTE ------------------

# nombre del archivo de la máscara de la cámara (blanco = cámara / NO imprimible)
CAMERA_MASK_NAME = "camera_mask.png"

from pathlib import Path

def ensure_camera_mask(base_path: Path) -> Path:
    """
    Busca la máscara de cámara localmente.
    - 1) ASSETS_DIR/CAMERA_MASK_NAME
    - 2) Mismo directorio que base_path
    Si no existe, lanza FileNotFoundError (lo atrapamos luego para no romper el flujo).
    """
    cand = [
        ASSETS_DIR / CAMERA_MASK_NAME,
        base_path.with_name(CAMERA_MASK_NAME),
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontró {CAMERA_MASK_NAME} en {ASSETS_DIR} ni junto a {base_path.name}")

def _grow_mask(mask_L: Image.Image, px: int = 6, blur: int = 2) -> Image.Image:
    """
    Expande la máscara 'px' píxeles (tolerancia) y suaviza el borde.
    Útil para dejar holgura alrededor de las lentes.
    """
    m = mask_L if mask_L.mode == "L" else mask_L.convert("L")
    if px > 0:
        m = m.filter(ImageFilter.MaxFilter(size=px*2+1))
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur))
    return m
# ------------------ HACER ESTUCHE CON IMAGEN------------------
# === Helpers / paths (arriba del archivo, junto a tus imports) ===

from fastapi import File, Form, UploadFile
from pathlib import Path

UPLOADS_DIR = Path("./media/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
# --- OpenAI images edit: compatibilidad con distintas versiones del SDK ---
from PIL import ImageStat
def _fit_to_mask_area(
    img_rgba: Image.Image,
    mask_L: Image.Image,
    W: int, H: int,
    margin: float = 0.04,
    mode: str = "contain",
    bleed_px: int = 8
) -> Image.Image:
    """
    Encaja la imagen dentro del bbox de la máscara:
    - mode='contain': respeta la imagen completa (puede dejar borde)
    - mode='cover' : llena el área (puede recortar, recomendado para imprimir)
    Aplica un pequeño 'bleed' (sangrado) para cubrir los bordes del estuche.
    """
    img = img_rgba.convert("RGBA")
    m = mask_L if mask_L.mode == "L" else mask_L.convert("L")

    # Binariza suave para obtener un bbox más estable (descarta ruido bajo)
    m_bin = m.point(lambda p: 255 if p > 8 else 0)
    bbox = m_bin.getbbox() or (0, 0, W, H)  # (left, top, right, bottom)

    left, top, right, bottom = bbox
    safe_w = max(1, right - left)
    safe_h = max(1, bottom - top)

    # margen interior en porcentaje del área segura
    pad_w = int(safe_w * max(0.0, min(0.15, margin)))
    pad_h = int(safe_h * max(0.0, min(0.15, margin)))
    tgt_w = max(1, safe_w - 2 * pad_w)
    tgt_h = max(1, safe_h - 2 * pad_h)

    # encaje
    ar_img = img.width / max(1, img.height)
    ar_box = tgt_w / max(1, tgt_h)

    if mode == "cover":
        # llena el área: escalado por el mayor eje y recorte centrado
        if ar_img > ar_box:
            new_h = tgt_h + bleed_px * 2  # añade bleed
            new_w = int(round(new_h * ar_img))
        else:
            new_w = tgt_w + bleed_px * 2
            new_h = int(round(new_w / ar_img))
        tmp = img.resize((new_w, new_h), Image.LANCZOS)

        # recorte al tamaño target + bleed (centrado)
        crop_w = tgt_w + bleed_px * 2
        crop_h = tgt_h + bleed_px * 2
        x0 = max(0, (new_w - crop_w) // 2)
        y0 = max(0, (new_h - crop_h) // 2)
        tmp = tmp.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        fitted = tmp
    else:
        # contain
        tmp = img.copy()
        tmp.thumbnail((tgt_w, tgt_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (tgt_w, tgt_h), (0, 0, 0, 0))
        cx = (tgt_w - tmp.width) // 2
        cy = (tgt_h - tmp.height) // 2
        canvas.alpha_composite(tmp, (cx, cy))
        fitted = canvas

    # pegar centrado dentro del bbox de la máscara
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    paste_x = left + (safe_w - fitted.width) // 2
    paste_y = top + (safe_h - fitted.height) // 2
    out.alpha_composite(fitted, (paste_x, paste_y))
    return out


def _image_edit(client, image_path: str, prompt: str, size: str, transparent: bool = True):
    """
    Intenta usar client.images.edits (plural). Si no existe, usa client.images.edit (singular).
    Si soporta background transparente, lo pide; si no, reintenta sin ese parámetro.
    """
    fn = getattr(client.images, "edits", None) or getattr(client.images, "edit", None)
    if fn is None:
        raise RuntimeError("El SDK de OpenAI no tiene images.edits ni images.edit")

    with open(image_path, "rb") as f:
        kwargs = dict(model="gpt-image-1", image=f, prompt=prompt, size=size)
        if transparent:
            try:
                return fn(**{**kwargs, "background": "transparent"})
            except Exception:
                # Algunas versiones no soportan 'background'
                return fn(**kwargs)
        return fn(**kwargs)
def _save_upload_sync(file: UploadFile) -> Path:
    """Guarda la imagen subida y devuelve la ruta local."""
    suffix = Path(file.filename or "upload").suffix or ".png"
    out = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    with out.open("wb") as f:
        f.write(file.file.read())
    return out

def _ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img

def _fit_with_margin(img: Image.Image, W: int, H: int, margin: float = 0.06) -> Image.Image:
    """Escala la imagen a un cuadro WxH dejando margen (%) para no cortar al imprimir."""
    max_w = int(W * (1 - margin))
    max_h = int(H * (1 - margin))
    img = img.copy()
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x = (W - img.width) // 2
    y = (H - img.height) // 2
    canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)
    return canvas
from PIL import ImageChops

def _autocrop_borders(img_rgba: Image.Image, tol: int = 4) -> Image.Image:
    """
    Recorta padding uniforme alrededor del motivo (muchos renders vienen con borde).
    """
    img = img_rgba.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        return img.crop(bbox)
    # fallback: por color del pixel (0,0)
    bg = Image.new("RGBA", img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg).convert("L")
    diff = diff.point(lambda p: 255 if p > tol else 0)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img

def _remove_flat_bg(img_rgba: Image.Image, tol: int = 16) -> Image.Image:
    """
    Vuelve transparente un fondo plano claro (blanco/gris) típico de algunos renders.
    """
    img = img_rgba.convert("RGBA")
    px = img.load()
    w, h = img.size
    r0, g0, b0, a0 = px[0, 0]
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if abs(r - r0) <= tol and abs(g - g0) <= tol and abs(b - b0) <= tol:
                px[x, y] = (r, g, b, 0)
    return img



# ------------------ Drive helpers ------------------
def _drive_client():
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    if not GOOGLE_CREDS_VALUE:
        raise RuntimeError("Falta GOOGLE_CREDS_JSON/GOOGLE_CREED_JSON")
    try:
        if GOOGLE_CREDS_VALUE.strip().startswith("{"):
            info = json.loads(GOOGLE_CREDS_VALUE)
            creds = Credentials.from_service_account_info(info, scopes=scopes)
        else:
            creds = Credentials.from_service_account_file(GOOGLE_CREDS_VALUE, scopes=scopes)
    except Exception as e:
        raise RuntimeError(f"Credenciales inválidas: {e}")
    return build("drive", "v3", credentials=creds)

def drive_list_children(parent_id: str, q_extra: str=""):
    svc = _drive_client()
    q = f"'{parent_id}' in parents and trashed=false"
    if q_extra: q = f"{q} and ({q_extra})"
    items, tok = [], None
    while True:
        res = svc.files().list(q=q, spaces="drive",
                               fields="files(id,name,mimeType),nextPageToken",
                               pageToken=tok).execute()
        items += res.get("files", []); tok = res.get("nextPageToken")
        if not tok: break
    return items

def drive_find_subfolder(parent_id: str, name: str):
    it = drive_list_children(parent_id, f"name='{name}' and mimeType='application/vnd.google-apps.folder'")
    return it[0] if it else None

def drive_find_file(folder_id: str, name: str):
    it = drive_list_children(folder_id, f"name='{name}'")
    return it[0] if it else None

def drive_download(file_id: str) -> bytes:
    svc = _drive_client()
    return svc.files().get_media(fileId=file_id).execute()
def ensure_base_from_drive(force: bool = False) -> pathlib.Path:
    """
    Asegura BASE_NAME en ASSETS_DIR.
    - Si force=True (o env DRIVE_FORCE_REFRESH=1), siempre re-descarga desde Drive.
    - Si no, devuelve el local si ya existe; si no existe, lo descarga.
    """
    local = ASSETS_DIR / BASE_NAME

    # Permite forzar por variable de entorno en deploy (p. ej. en Render/Heroku)
    try:
        env_force = str(os.getenv("DRIVE_FORCE_REFRESH", "")).strip() in ("1", "true", "True")
    except Exception:
        env_force = False
    force = bool(force or env_force)

    if not (DRIVE_ROOT_ID and GOOGLE_CREDS_VALUE):
        raise RuntimeError("No hay base.png local y Drive no está configurado")

    # Si no forzamos y ya existe, devolvemos tal cual
    if local.exists() and not force:
        return local

    # Descarga desde Drive (siempre que falte o estemos forzando)
    sub = drive_find_subfolder(DRIVE_ROOT_ID, SUBFOLDER_NAME)
    if not sub:
        raise RuntimeError(f"No se encontró subcarpeta '{SUBFOLDER_NAME}'")
    f = drive_find_file(sub["id"], BASE_NAME)
    if not f:
        raise RuntimeError(f"No se encontró {BASE_NAME} en Drive")

    data = drive_download(f["id"])
    local.write_bytes(data)
    return local
def ensure_camaras_from_drive(force: bool = False) -> pathlib.Path:
    """
    Asegura CAMARAS_NAME en ASSETS_DIR.
    - Si force=True (o env DRIVE_FORCE_REFRESH=1), siempre re-descarga desde Drive.
    - Si no, devuelve el local si ya existe; si no existe, lo descarga.
    """
    local = ASSETS_DIR / CAMARAS_NAME

    # Permite forzar por variable de entorno en deploy (p. ej. en Render/Heroku)
    try:
        env_force = str(os.getenv("DRIVE_FORCE_REFRESH", "")).strip() in ("1", "true", "True")
    except Exception:
        env_force = False
    force = bool(force or env_force)

    if not (DRIVE_ROOT_ID and GOOGLE_CREDS_VALUE):
        raise RuntimeError("No hay camaras.png local y Drive no está configurado")

    # Si no forzamos y ya existe, devolvemos tal cual
    if local.exists() and not force:
        return local

    # Descarga desde Drive (siempre que falte o estemos forzando)
    sub = drive_find_subfolder(DRIVE_ROOT_ID, SUBFOLDER_NAME)
    if not sub:
        raise RuntimeError(f"No se encontró subcarpeta '{SUBFOLDER_NAME}'")
    f = drive_find_file(sub["id"], CAMARAS_NAME)
    if not f:
        raise RuntimeError(f"No se encontró {CAMARAS_NAME} en Drive")

    data = drive_download(f["id"])
    local.write_bytes(data)
    return local


# ------------------ Base & máscara ------------------
def load_rgba(p: pathlib.Path) -> Image.Image:
    return Image.open(p).convert("RGBA")

def ensure_mask(base_path: pathlib.Path) -> pathlib.Path:
    mask_path = ASSETS_DIR / "base_mask.png"
    if mask_path.exists(): return mask_path
    img = cv2.imdecode(np.frombuffer(base_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError("No se pudo leer base.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low, high = np.array([0,0,0], np.uint8), np.array([180,60,80], np.uint8)
    mask = cv2.inRange(hsv, low, high)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: raise RuntimeError("No se detectó el estuche en base.png")
    cnt = max(cnts, key=cv2.contourArea)
    canvas = np.zeros_like(mask)
    cv2.drawContours(canvas, [cnt], -1, 255, thickness=-1)
    kernel = np.ones((9,9), np.uint8)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel, iterations=2)
    canvas = cv2.GaussianBlur(canvas, (9,9), 0)
    ok = cv2.imencode(".png", canvas)[1]
    mask_path.write_bytes(ok.tobytes())
    return mask_path

def paste_with_mask(base_rgba: Image.Image, content_rgba: Image.Image, mask_gray: Image.Image) -> Image.Image:
    if mask_gray.mode != "L": mask_gray = mask_gray.convert("L")
    content_rgba = content_rgba.resize(base_rgba.size, Image.LANCZOS)
    mask_gray    = mask_gray.resize(base_rgba.size, Image.LANCZOS)
    out = base_rgba.copy()
    out.paste(content_rgba, (0,0), mask_gray)
    return out





# ------------------ Modelos conversación ------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatReq(BaseModel):
    history: List[ChatMessage] = []
    state: Dict[str, Any] | None = None


ASSISTANT_SYSTEM = """
Eres el asesor de RAVIE. Hablas en español de Colombia, cercano y profesional.
No pidas el modelo del teléfono; el preview siempre es sobre un mockup base.
Aclara cuando corresponda: “El preview es genérico, pero el estuche final se fabrica exactamente para tu modelo.”

== OBJETIVO ==
Guiar al cliente para crear un diseño de estuche: recoger tema/protagonistas, estilo, colores y detalles; generar preview; ajustar cambios; pasar a checkout.

== FASES ==
- pidiendo_tema -> pregunta qué quiere (personaje/tema/foto)
- pidiendo_estilo -> pregunta estilo y colores (y texto opcional)
- listo_para_generar -> anuncia que mostrarás un preview y solicita la tool
- generando_preview -> espera resultado de tool
- ajustando -> pregunta si desea cambios y regenera si hace falta
- checkout -> confirma pedido y pide datos de envío

== GUÍA DE ESTILO Y VOCABULARIO ==
- Tono: cálido, breve, positivo, orientado a acción.
- Vocabulario SI:
  - “Listo”, “Perfecto”, “De una”, “Súper”, “Te muestro”, “¿Quieres algún cambio?”
  - “Podemos ajustar color/fondo/estilo/texto”
  - “El preview es genérico; el final llega para tu modelo exacto”
- Vocabulario NO:
  - Evita tecnicismos largos o fríos (“procederemos a…”, “estimado cliente…”)
  - Evita prometer imposibles (“entrega inmediata”, “gratis siempre”)
- Emojis: máximo 1–2 por mensaje (👍✨🎨📦). Nunca pongas más de 2.
- Estructura de cada respuesta:
  1) Afirmar/confirmar en 1 línea (máx. ~20–30 palabras)
  2) Si aplica, 1 frase de valor/claridad (ej. el disclaimer del mockup)
  3) Cerrar con UNA pregunta clara para avanzar (solo una pregunta por turno)
- Longitud: 1–3 líneas, no más.
- Si el usuario hace una pregunta general (FAQ), respóndela y ofrece seguir con el diseño (“¿Quieres que lo armemos con X estilo?”).
- Si menciona cambios específicos (ej. “fondo negro”, “Super Saiyajin 2”, “brillante”), regístralos en detalles del brief.

== CONTENIDO DEL BRIEF ==
- tema
- protagonistas (opcional)
- estilo (manga, realista, minimalista, neón, retro…)
- colores
- texto (opcional)
- detalles_extra (lista: ej. “fondo negro”, “SSJ2”, “acabado brillante”)

== POLÍTICAS DE GENERACIÓN ==
- Nunca generes imagen en el primer turno.
- Solo genera cuando haya (tema o protagonistas) Y estilo.
- Al generar, di explícitamente: “Te muestro un preview… (recuerda: es genérico; el final es para tu modelo exacto)”.
- Tras mostrar preview: “¿Quieres algún cambio o lo dejamos así?”

== MANEJO DE FAQs (respuestas breves) ==
- Compatibilidad: “Sí, trabajamos tu modelo exacto. El preview es genérico, pero el final llega para tu modelo.”
- Precio: “Desde $59.900 COP según acabado. ¿Quieres que lo armemos y te muestro?”
- Envío: “2–5 días hábiles según ciudad. ¿Seguimos con el diseño?”
- Material: “Resistente anti-golpes, opción mate o brillante. ¿Cuál prefieres?”
(Tras responder la FAQ, haz UNA pregunta para avanzar el diseño.)

== SALIDA JSON OBLIGATORIA ==
Devuelve SOLO JSON:
{
  "reply": "texto breve siguiendo la guía de estilo",
  "brief": {
    "tema": "...",
    "protagonistas": "...",
    "estilo": "...",
    "colores": "...",
    "texto": "...",
    "detalles_extra": ["..."]
  },
  "phase": "pidiendo_tema | pidiendo_estilo | listo_para_generar | generando_preview | ajustando | checkout",
  "action": null | { "name": "generate", "prompt": "prompt para gpt-image-1", "final": false }
}
"""

def _prompt_from_brief(brief: Dict[str, Any]) -> str:
    tema    = brief.get("tema") or brief.get("protagonistas") or "diseño abstracto moderno"
    estilo  = brief.get("estilo")
    colores = brief.get("colores")
    texto   = brief.get("texto")
    parts = [f"Phone case art with {tema}", "vertical composition", "clean layout", "high detail", "no borders"]
    if estilo:  parts.append(f"style {estilo}")
    if colores: parts.append(f"palette {colores}")
    if texto:   parts.append(f"include the word '{texto}' with aesthetic typography")
    return ", ".join(parts)

# ------------------ SUBIR IMAGEN CON COPY ------------------

@router.post("/chat-upload")
async def chat_upload(
    payload: str = Form(...),               # JSON del ChatReq como string
    file: UploadFile | None = File(None),   # imagen opcional
):
    data = json.loads(payload)
    req = ChatReq(**data)                   # tu mismo modelo

    # Inicializa state y guarda imagen si viene
    req.state = _init_state(req.state)
    if file:
        path = _save_upload_sync(file)
        req.state["imagen_usuario"] = str(path)

    # Reusa tu pipeline normal:
    return chat(req)  # 👈 llama al mismo handler que ya tienes





# ------------------ Chat ------------------
@router.post("/chat")   # 👈 usar router, no app
def chat(req: ChatReq):
    t0 = time.time()
    state = _init_state(req.state)

    # Asegura flags de agente (por si el state viene viejo)
    state.setdefault("asked_preview", False)   # legacy (no se usa)
    state.setdefault("confirm_preview", False) # legacy (no se usa)

    # 0) Primer turno (sin LLM)
    if not req.history:
        _set_fase(state, "pidiendo_tema")
        reply = ("¡Hola! 👋 Cuéntame qué quieres en tu estuche (personaje/tema) y el estilo/colores. "
                 "El preview es genérico, pero el estuche final va exactamente para tu modelo.")
        return _ok(reply, state, None)

    # 0.5) Respuesta rápida a FAQs sin pasar por el LLM
    last_user = next((m.content for m in reversed(req.history) if m.role == "user"), "")
    canned = quick_faq_reply(last_user) if 'quick_faq_reply' in globals() else None
    if canned:
        return _ok(canned, state, None)

    # 1) Cliente OpenAI
    try:
        client = make_openai_client()
    except Exception as e:
        log.exception("[/chat] No se pudo crear cliente OpenAI")
        friendly = ("Tuve un problema técnico procesando tu idea 😅. "
                    "Sigamos: dime el tema/personaje y el estilo/colores que quieres.")
        return _ok(friendly, state, None)

    # 2) Construir mensajes (incluye fase/brief actual)
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM}]
    messages.append({"role": "user", "content": "STATE:\n" + json.dumps(state, ensure_ascii=False)})
    for m in req.history[-14:]:
        messages.append(m.model_dump())

    # 3) LLM -> JSON (con fallback si hay mismatched_organization)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.5,
            messages=messages,
        )
    except AuthenticationError as e:
        if "mismatched_organization" in str(e).lower():
            log.warning("[/chat] mismatched_organization → retry sin OpenAI-Organization")
            client2 = make_openai_client_no_org()
            resp = client2.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                temperature=0.5,
                messages=messages,
            )
        else:
            log.exception("[/chat] AuthenticationError")
            friendly = ("Tuve un problema técnico procesando tu idea 😅. "
                        "Sigamos: dime el tema/personaje y el estilo/colores que quieres.")
            return _ok(friendly, state, None)
    except Exception as e:
        log.exception("[/chat] error LLM")
        friendly = ("Se me presentó un fallo al pensar tu diseño. "
                    "Cuéntame el tema/personaje y el estilo/colores y seguimos.")
        return _ok(friendly, state, None)

    # 3.1) Parseo de JSON del LLM (si falla, usa dict vacío)
    data = {}
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        log.exception("[/chat] JSON inválido desde el LLM")
        data = {}

    reply   = (data.get("reply") or "").strip()
    brief   = data.get("brief") or {}
    phase_s = (data.get("phase") or "").strip()

    # 4) Actualiza estado con lo que vino del LLM
    for k in ["tema","protagonistas","estilo","colores","texto","detalles_extra"]:
        if brief.get(k) not in (None, "", []):
            state[k] = brief[k]

    # 5) Resolver fase localmente (sin modelo) — con checkpoint_preview y defaults
    reply = (reply or "")  # cinturón: jamás None

    if phase_s in FASES:
        _set_fase(state, phase_s)
    else:
        tema_o_prot = (state.get("tema") or state.get("protagonistas"))
        if not tema_o_prot:
            _set_fase(state, "pidiendo_tema")
        else:
            # Defaults de estilo/colores si faltan
            if not state.get("estilo"):
                state["estilo"] = "dinámico y colorido"
            if not state.get("colores"):
                tema_low = f"{state.get('tema','')} {state.get('protagonistas','')}".lower()
                if any(k in tema_low for k in ["rayo mcqueen", "rayo", "cars", "lightning mcqueen"]):
                    state["colores"] = "rojo con acentos amarillo/blanco"

            # Si no hay preview previo, vamos a checkpoint; si ya hubo, a ajustar
            if state.get("ultimo_preview") is None:
                _set_fase(state, "checkpoint_preview")
            else:
                _set_fase(state, "ajustando")

    # === 5.1) Checkpoint: decidir 'generar ya' o 'aplicar y generar' (SIN pedir confirmación extra) ===
    if state.get("fase") == "checkpoint_preview":
        txt = (last_user or "").strip().lower()

        GENERAR_YA = [
            "hazlo así", "hazlo asi", "dale", "genéralo", "generalo",
            "ok", "listo", "tal cual", "como está", "como esta",
            "sí", "si", "hágale", "hagale"
        ]
        CLAVES_CAMBIO = [
            "poner", "agregar", "añadir", "cambiar", "color", "colores",
            "nombre", "texto", "fondo", "brillo", "mover", "centrar",
            "borde", "bordes", "tipografía", "tipografia", "logo"
        ]

        def contiene_alguna(pats, s): 
            return any(p in s for p in pats)

        if contiene_alguna(GENERAR_YA, txt) or txt in ["ok", "si", "sí", "listo"]:
            _set_fase(state, "listo_para_generar")
        elif contiene_alguna(CLAVES_CAMBIO, txt):
            if "nombre" in txt and not state.get("texto"):
                try:
                    after = txt.split("nombre", 1)[1].strip(": ,.-")
                    for cut in [" y ", " con ", " en ", " fondo ", " color "]:
                        if cut in after:
                            after = after.split(cut, 1)[0].strip()
                    if after:
                        state["texto"] = after.upper()[:20]
                except Exception:
                    pass
            _set_fase(state, "listo_para_generar")
        else:
            _set_fase(state, "listo_para_generar")

        if not reply:
            reply = ("Perfecto, te muestro un preview… "
                     "(recuerda: es genérico; el final va exacto a tu modelo)")

    preview_url = None  # SIEMPRE inicializado

# 6) Generación directa (SIN confirm_preview). Basta con estar en 'listo_para_generar'.
    if _brief_minimo(state) and state["fase"] == "listo_para_generar":
        _set_fase(state, "generando_preview")
        try:
            base_path = ASSETS_DIR / BASE_NAME
            if not base_path.exists():
                base_path = ensure_base_from_drive()

            # base y máscara general del estuche
            mask_path = ensure_mask(base_path)
            base = load_rgba(base_path)
            mask = Image.open(mask_path).convert("L")

            # --- CÁMARA + MÁSCARAS ROBUSTAS ---
            # 1) binariza estuche para bbox nítido
            mask_bin = mask.point(lambda p: 255 if p > 8 else 0).convert("L")

            # 2) intenta cargar máscara de cámara (tolerante)
            cam = None
            try:
                try:
                    camera_mask_path = ensure_camera_mask(base_path)  # si existe helper
                except NameError:
                    camera_mask_path = ASSETS_DIR / "camera_mask.png"
                if camera_mask_path and Path(camera_mask_path).exists():
                    cam = Image.open(camera_mask_path).convert("L")
            except Exception:
                cam = None

            if cam is not None:
                # binariza cámara (blanco = NO imprimible), holgura alrededor de lentes
                cam_bin = cam.point(lambda p: 255 if p > 8 else 0).convert("L")
                cam_bin_grown = _grow_mask(cam_bin, px=6, blur=2)
                # imprimible = estuche - cámara
                print_mask = ImageChops.subtract(mask_bin, cam_bin_grown, scale=1.0, offset=0)
                log.info(f"[/chat] camera mask aplicada: {camera_mask_path}")
            else:
                log.warning("[/chat] camera mask no disponible → imprimible = mask total")
                print_mask = mask_bin
                cam_bin_grown = None  # para recorte posterior opcional

            W, H = base.size

            # --- BLEED/SANGRADO ---
            BLEED_PX = 8  # 6–12 según tu impresora
            print_mask_bleed = _grow_mask(print_mask, px=BLEED_PX, blur=2)

            # Re-cortar la cámara DESPUÉS del bleed (clave para no invadir la ventana)
            if cam_bin_grown is not None:
                cam_bleed = _grow_mask(cam_bin_grown, px=BLEED_PX + 2, blur=2)
                print_mask_bleed = ImageChops.subtract(print_mask_bleed, cam_bleed, scale=1.0, offset=0)

            # Log útil para debug
            try:
                log.info(f"[/chat] base={base.size} mask_bbox={mask_bin.getbbox()} print_bbox={print_mask.getbbox()}")
            except Exception:
                pass

        except Exception as e:
            log.exception("[/chat] Error preparando lienzo")
            return _fail(f"❌ No pude preparar el lienzo del estuche: {e}", state)

        # Prompt y calidad (low por defecto; medium si es final)
        prompt = _prompt_from_brief(state)[:800]
        is_final = bool(state.get("final"))
        img_quality = "medium" if is_final else "low"

        # Si el tema parece de marca/IP, prepara un prompt seguro por si toca reintentar
        _topic = f"{state.get('tema','')} {state.get('protagonistas','')}".lower()
        _brand_terms = [
            "rayo mcqueen","lightning mcqueen","cars","disney","pixar",
            "goku","gohan","vegeta","dragon ball","marvel","dc","batman","spiderman",
        ]
        safe_prompt = (
            "Convierte la imagen/idea en arte ORIGINAL listo para impresión: "
            "fondo totalmente transparente, bordes nítidos, alto contraste, "
            "motivo centrado, margen seguro 6%, sin logos ni marcas registradas."
        )
        if any(t in _topic for t in _brand_terms):
            safe_prompt = safe_prompt[:800]

        # === Si el usuario subió imagen, preferimos 'edits' con fondo transparente ===
        user_img_path = state.get("imagen_usuario")  # p.ej. lo llena /chat-upload
        gen_size = "1024x1024" if not is_final else "1536x1536"

        img = None
        try:
            # cliente SIN organization para evitar 401 en images.*
            gen_client = make_openai_client_no_org()

            if user_img_path:
                prompt_upload = (
                    f"Estilo: {state.get('estilo','vibrante y limpio')}. "
                    f"Colores: {state.get('colores','equilibrados, alto contraste')}. "
                    "Diseña un arte listo para impresión de estuche; fondo TRANSPARENTE; "
                    "bordes/contornos limpios; motivo centrado; margen 6%; "
                    "evita texto pequeño, marcas de agua y logos."
                )[:800]
                img = _image_edit(gen_client, user_img_path, prompt_upload, gen_size, transparent=True)
            else:
                img = gen_client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    size=gen_size,
                    quality=img_quality,
                )

        except AuthenticationError as e:
            if "mismatched_organization" in str(e).lower():
                log.warning("[/chat] images.* mismatched_organization → retry sin organization")
                gen_client = make_openai_client_no_org()
                if user_img_path:
                    img = _image_edit(gen_client, user_img_path, prompt_upload, gen_size, transparent=True)
                else:
                    img = gen_client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt,
                        size=gen_size,
                        quality=img_quality,
                    )
            else:
                raise
        except Exception as e:
            msg = str(e).lower()
            if "moderation" in msg or "rejected by the safety system" in msg or "bad request" in msg:
                log.warning("[/chat] moderation/rechazo → reintento con prompt seguro (sin marcas)")
                try:
                    gen_client = make_openai_client_no_org()
                    if user_img_path:
                        img = _image_edit(gen_client, user_img_path, safe_prompt[:800], gen_size, transparent=True)
                        reply = (reply + "\n\nHice un preview inspirado en tu imagen, sin logos ni marcas.").strip()
                    else:
                        img = gen_client.images.generate(
                            model="gpt-image-1",
                            prompt=safe_prompt[:800],
                            size=gen_size,
                            quality=img_quality,
                        )
                        reply = (reply + "\n\nHice un preview inspirado en tu idea, sin logos ni marcas registradas.").strip()
                except Exception:
                    log.exception("[/chat] Error generando luego de sanitizar prompt")
                    _set_fase(state, "pidiendo_estilo")
                    reply = (reply + "\n\n❌ No pude generar el preview con ese tema exacto. "
                                   "Puedo hacer una versión original (inspirada) o cambiamos colores/estilo.").strip()
                    img = None
            else:
                log.exception("[/chat] Error generando (intento 1)")
                _set_fase(state, "pidiendo_estilo")
                reply = (reply + "\n\n❌ No pude generar el preview ahora. "
                               "Cuéntame si cambiamos color/fondo/estilo y reintento.").strip()
                img = None

        # Si hay imagen, procesamos; si no, dejamos reply con el mensaje y salimos
        if img:
            try:
                b64 = img.data[0].b64_json
                content = Image.open(io.BytesIO(base64.b64decode(b64)))
                content_rgba = _ensure_rgba(content)

                # 1) recorta padding del render (si vino con borde)
                content_rgba = _autocrop_borders(content_rgba)

                # 2) si vino con fondo plano (blanco/gris), hazlo transparente
                content_rgba = _remove_flat_bg(content_rgba, tol=20)

                # 3) Encaja usando la zona imprimible real (estuche sin cámara)
                try:
                    content_rgba = _fit_to_mask_area(
                        content_rgba, print_mask, W, H,
                        margin=0.04, mode="cover", bleed_px=BLEED_PX
                    )
                except TypeError:
                    content_rgba = _fit_to_mask_area(
                        content_rgba, print_mask, W, H,
                        margin=0.04, mode="cover"
                    )

                # 4) Composición inicial (arte sobre base, respetando máscara + bleed)
                out = Image.composite(content_rgba, base, print_mask_bleed)

                # 5) Overlay de cámaras (traído desde Drive)
                try:
                    cam_overlay_path = ensure_camera_overlay_from_drive()
                    cam_overlay = Image.open(cam_overlay_path).convert("RGBA")

                    # Ajustamos al mismo tamaño del base
                    if cam_overlay.size != base.size:
                        cam_overlay = cam_overlay.resize(base.size, Image.LANCZOS)

                    # Pegamos cámaras encima (se ven siempre las lentes reales)
                    out.alpha_composite(cam_overlay, (0, 0))
                    log.info(f"[/chat] overlay de cámaras aplicado desde {cam_overlay_path}")
                except Exception as e:
                    log.warning(f"[/chat] no se pudo aplicar overlay de cámaras: {e}")

                # 6) Guardar preview
                out_id = f"{uuid.uuid4()}.png"
                out_path = PREVIEWS_DIR / out_id
                out.save(out_path, "PNG")

                preview_url = f"/media/{out_id}"
                state["ultimo_preview"] = preview_url
                log.info(f"[/chat] preview listo: {preview_url} -> {out_path}")

                _set_fase(state, "ajustando")
                state["confirm_preview"] = False
                state["asked_preview"] = False

                tail = "" if is_final else " (preview en calidad económica)"
                extra = "Aquí está tu preview 👇 ¿Quieres algún cambio o lo dejamos así?" + tail
                reply = (reply + "\n\n" + extra).strip()
            except Exception as e:
                log.exception("[/chat] Error procesando imagen generada")
                _set_fase(state, "pidiendo_estilo")
                reply = (reply + f"\n\n❌ La generación se completó pero falló el post-proceso: {e}").strip()






# 7) Si el usuario confirma → checkout
    if state["fase"] == "ajustando":
        last_low = (last_user or "").lower()
        if any(x in last_low for x in [
            "usar este diseño","quiero ese","me gusta","agregar","añadir",
            "comprar","está bien","esta bien","déjalo así","dejalo asi","asi esta","así está",
            "dejar así","dejar asi","déjalo asi","dejalo así"
        ]):
            _set_fase(state, "checkout")
            reply = "Perfecto ✅ ¿Lo agrego al carrito? Necesito tu nombre, teléfono y ciudad para el envío."

# --- Fallback de preview antes de log y return ---
    if 'preview_url' not in locals() or not isinstance(preview_url, str):
        preview_url = state["ultimo_preview"] if isinstance(state.get("ultimo_preview"), str) else None

# Log de salida (opcional para debug)
    log.info(f"RESP /chat -> ok=True, fase={state.get('fase')}, reply_len={len(reply or '')}, preview={preview_url}")

# Return final SIEMPRE en JSON
    reply = reply or "Listo, cuéntame el tema del diseño y te lo armo."
    if not isinstance(preview_url, str):
        preview_url = None
    return _ok(reply, state, preview_url)



@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html><html lang="es"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Asistente RAVIE</title>
<style>
*{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Roboto}
.wrap{display:flex;flex-direction:column;height:100vh}
.header{padding:14px 16px;font-weight:800;border-bottom:1px solid #eee}
.chat{flex:1;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:12px;background:#fff}
.msg{max-width:92%;padding:10px 12px;border-radius:14px;border:1px solid #eee;box-shadow:0 1px 0 rgba(0,0,0,.02)}
.me{align-self:flex-end;background:#111;color:#fff;border-color:#111}
.bot{align-self:flex-start;background:#f8fafc}
.muted{color:#6b7280;font-size:13px}
.img{display:block;max-width:100%;border-radius:12px;border:1px solid #eee}
.footer{padding:10px;border-top:1px solid #eee;display:flex;gap:8px;align-items:center}
.footer input[type=text]{flex:1;padding:12px;border-radius:10px;border:1px solid #d1d5db}
.btn{appearance:none;border:0;border-radius:10px;padding:12px 16px;font-weight:700;cursor:pointer}
.btn-dark{background:#111;color:#fff}
.btn-ghost{background:#fff;border:1px solid #d1d5db}
.actions{display:flex;gap:8px;margin-top:8px}

/* BOTÓN ADJUNTAR MUY VISIBLE */
#attach{display:inline-flex;align-items:center;gap:8px;
  padding:12px 14px;border-radius:10px;border:2px solid #111;background:#fff;color:#111;font-weight:800}
#attach .ic{font-size:16px}
#attach:hover{background:#111;color:#fff}
#attach:disabled{opacity:.6;cursor:not-allowed}

/* Chip del nombre de archivo */
.chip{margin:6px 10px;display:none;align-items:center;gap:8px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:999px;padding:4px 10px;font-size:12px;color:#374151}
.chip .x{cursor:pointer;font-weight:700}
</style></head><body>
<div class="wrap">
  <div class="header">Asistente RAVIE</div>
  <div id="chat" class="chat"></div>

  <div id="chip" class="chip"><span id="chipName"></span><span id="chipClear" class="x">×</span></div>

  <div class="footer">
    <button id="attach" title="Adjuntar imagen"><span class="ic">📎</span> Adjuntar</button>
    <input id="file" type="file" accept="image/*" style="display:none"/>
    <input id="text" type="text" placeholder="Dime el modelo, tema/protagonistas, estilo, color…" />
    <button id="send" class="btn btn-dark">Enviar</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat'); const $=s=>document.querySelector(s);
let history=[], state={};

function esc(s){return (s||'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}
function addMsg(h,who='bot'){const d=document.createElement('div');d.className='msg '+(who==='me'?'me':'bot');d.innerHTML=h;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;}
function addTyping(){return addMsg('<span class="muted">escribiendo…</span>','bot');}

addMsg('<b>¡Hola!</b> Soy tu diseñador RAVIE. ¿Para qué <b>modelo de teléfono</b> quieres tu estuche hoy?', 'bot');

const attach = $('#attach'), file=$('#file'), chip=$('#chip'), chipName=$('#chipName'), chipClear=$('#chipClear');
let pendingFile=null;

attach.addEventListener('click', ()=> file.click());
file.addEventListener('change', ()=>{
  pendingFile = file.files && file.files[0] ? file.files[0] : null;
  if(pendingFile){ chipName.textContent='Imagen: '+pendingFile.name; chip.style.display='inline-flex'; }
  else{ chip.style.display='none'; }
});
chipClear.addEventListener('click', ()=>{ pendingFile=null; file.value=''; chip.style.display='none'; });

async function sendTurn(text){
  if(text){ history.push({role:'user',content:text}); addMsg(esc(text),'me'); }
  if(!text && pendingFile){ addMsg('📎 <i>Imagen adjunta: '+esc(pendingFile.name)+'</i>','me'); history.push({role:'user',content:'[Imagen adjunta]'}); }

  const t=addTyping();
  try{
    let r;
    if(pendingFile){
      const fd=new FormData();
      fd.append('payload', JSON.stringify({history,state}));
      fd.append('file', pendingFile);
      r = await fetch('/chat-upload', { method:'POST', body:fd });
    }else{
      r = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({history,state}) });
    }
    const j=await r.json(); t.remove();
    if(!j.ok) throw new Error(JSON.stringify(j));
    addMsg(esc(j.reply),'bot');
    state=j.state||state; history.push({role:'assistant',content:j.reply});

    pendingFile=null; file.value=''; chip.style.display='none';

    if(j.preview_url){
      const abs=new URL(j.preview_url, location.href).href;
      const el=addMsg('<img class="img" src="'+abs+'"/><div class="actions"><button id="use" class="btn btn-dark">Usar este diseño</button><button id="edit" class="btn btn-ghost">Cambiar algo</button></div>','bot');
      el.querySelector('#use').addEventListener('click',()=> parent.postMessage({type:'RAVIE_USE_DESIGN',img:abs,modelo:state.modelo_telefono||'Personalizado IA',precio:'59900'},'*'));
      el.querySelector('#edit').addEventListener('click',()=> $('#text').focus());
    }
  }catch(e){ t.remove(); addMsg('❌ <b>Error</b> <span class="muted">'+e.message+'</span>','bot'); }
}

const sendBtn=document.getElementById('send'); let sending=false;
sendBtn.addEventListener('click', async ()=>{
  if(sending) return;
  const tx=$('#text').value.trim();
  if(!tx && !pendingFile){ $('#text').focus(); return; }
  sending=true; sendBtn.disabled=true; const old=sendBtn.textContent; sendBtn.textContent='Procesando…';
  $('#text').value=''; await sendTurn(tx);
  sending=false; sendBtn.disabled=false; sendBtn.textContent=old;
});
document.getElementById('text').addEventListener('keydown', e=>{ if(e.key==='Enter'){ sendBtn.click(); }});
</script>
</body></html>
"""



# ------------------ Generación directa (si quieres llamarla sin chat) ------------------
class GenReq(BaseModel):
    prompt: str
    w: Optional[int] = None
    h: Optional[int] = None


# ===== Switch & saneamiento para /gen =====
ENABLE_DIRECT_GEN = os.getenv("ENABLE_DIRECT_GEN", "false").lower() == "true"
MIN_PROMPT_LEN = int(os.getenv("MIN_PROMPT_LEN", "12"))

@app.post("/gen")
def gen(req: GenReq):
    # ---- Gate: deshabilitado por defecto (usar /chat) ----
    if not ENABLE_DIRECT_GEN:
        raise HTTPException(403, "Direct generation is disabled. Usa el chat (/chat) para conversar y generar el preview.")

    # --- Log de entrada
    log.info("[/gen] IN prompt=%r w=%s h=%s", (req.prompt or "")[:80], req.w, req.h)

    # --- Saneamiento del prompt (evita gastos por mensajes triviales)
    prompt = (req.prompt or "").strip()
    if len(prompt) < MIN_PROMPT_LEN:
        raise HTTPException(400, f"Prompt demasiado corto ({len(prompt)} chars). Usa el chat para detallar tu idea.")

    # --- Cliente OpenAI (NO forzar organization si no corresponde)
    try:
        client = make_openai_client()
    except Exception as e:
        log.exception("[/gen] No se pudo crear cliente OpenAI")
        raise HTTPException(500, f"Config OpenAI inválida: {e}")

    # --- Asegurar base & máscara
    base_path = ASSETS_DIR / BASE_NAME
    if not base_path.exists():
        try:
            base_path = ensure_base_from_drive()
        except Exception as e:
            log.exception("[/gen] No hay base.png local y falló Drive")
            raise HTTPException(500, f"No existe base.png (ni Drive configurado): {e}")

    try:
        mask_path = ensure_mask(base_path)
        base = load_rgba(base_path)
        mask = Image.open(mask_path).convert("L")
    except Exception as e:
        log.exception("[/gen] Error preparando máscara/base")
        raise HTTPException(500, f"Error preparando máscara/base: {e}")

    # --- Tamaño destino
    W = req.w or base.width
    H = req.h or base.height
    log.info("[/gen] base_size=%sx%s target_size=%sx%s", base.width, base.height, W, H)

    # --- Generar imagen IA (barato por defecto)
    prompt = prompt[:800]
    log.info("[/gen] Llamando OpenAI images.generate size=1024x1024 quality=low")
    try:
        img = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="low",     # <<< reduce costo 4x para previews
        )
        b64 = img.data[0].b64_json
        content_rgba = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    except PermissionDeniedError:
        log.exception("[/gen] 403 PermissionDenied (org sin acceso)")
        raise HTTPException(
            403,
            "Tu cuenta/org no tiene acceso a gpt-image-1. Usa el chat o revisa tu organización en OpenAI."
        )
    except AuthenticationError:
        log.exception("[/gen] 401 AuthenticationError (key/org)")
        raise HTTPException(401, "API key inválido o no coincide con la organización asociada.")
    except APIError as e:
        log.exception("[/gen] APIError de OpenAI")
        raise HTTPException(502, f"Error temporal en OpenAI: {e}")
    except Exception as e:
        log.exception("[/gen] Error inesperado generando imagen")
        raise HTTPException(500, f"Error generando imagen: {e}")

    # --- Componer en el estuche
    try:
        if (W, H) != (base.width, base.height):
            base = base.resize((W, H), Image.LANCZOS)
            mask = mask.resize((W, H), Image.LANCZOS)

        out = paste_with_mask(base, content_rgba.resize((W, H), Image.LANCZOS), mask)
        out_id = f"{uuid.uuid4()}.png"
        out_path = PREVIEWS_DIR / out_id
        out.save(out_path, "PNG")
        log.info("[/gen] OK -> %s", out_path)
        return {"ok": True, "preview_url": f"/media/{out_id}", "quality": "low"}
    except Exception as e:
        log.exception("[/gen] Error componiendo/salvando imagen final")
        raise HTTPException(500, f"Error componiendo/salvando imagen: {e}")

# ------------------ Utilidades ------------------
@app.get("/media/{name}")
def media(name: str):
    p = PREVIEWS_DIR / name
    if not p.exists(): raise HTTPException(404, "No encontrado")
    return FileResponse(p, headers={"Cache-Control":"public, max-age=86400"})

@app.get("/healthz")
def healthz():
    return {"ok": True, "base_exists": (ASSETS_DIR/BASE_NAME).exists(), "org": bool(OPENAI_ORG_ID)}

@app.get("/")
def root():
    return {"ok": True, "service": "agente-ravie", "ui": "/ui"}
