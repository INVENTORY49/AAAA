# inventory.py — FastAPI mínimo con diagnóstico y home embebido
import os, sys, time, logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from urllib.parse import quote

APP_NAME = "Inventory demo (catch-all)"

# Huella al importar
print(">>> [inventory] importado desde:", __file__, "  cwd:", os.getcwd(), "  sys.path[0]:", sys.path[0], flush=True)

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
@app.middleware("http")
async def _security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline';"
    )
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("inventory")

@app.on_event("startup")
async def _startup():
    log.warning("🚀 Startup con %d rutas", len(app.router.routes))
    for r in app.router.routes:
        methods = ",".join(sorted(getattr(r, "methods", []) or []))
        log.warning("   -> %s %s", methods, r.path)

@app.middleware("http")
async def _log_req(request: Request, call_next):
    t0 = time.time()
    try:
        resp = await call_next(request)
    except Exception:
        log.exception("Unhandled %s %s", request.method, request.url.path)
        raise
    dt = (time.time() - t0) * 1000
    log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, getattr(resp, "status_code", "?"), dt)
    return resp

@app.exception_handler(StarletteHTTPException)
async def _http_exc(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        paths = [r.path for r in app.router.routes]
        log.warning("🚫 404 en %s. Rutas=%s", request.url.path, paths)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


# ---------- PROCESO DE VINCULACION DE INVENTARIO  ----------
import os, re, json, base64
from fastapi import HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from fastapi.responses import JSONResponse

SHEETS_SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

def _load_sa_info() -> dict:
    """
    Carga credenciales desde variables de entorno en este orden:
      1) GOOGLE_CREDS_JSON          (ruta a archivo o JSON plano)
      2) GOOGLE_CREDS_JSON_B64      (JSON en Base64)
      3) GOOGLE_SERVICE_ACCOUNT_JSON (compatibilidad: ruta o JSON)
      4) GOOGLE_SERVICE_ACCOUNT_JSON_B64 (compatibilidad: Base64)
    """
    raw = os.getenv("GOOGLE_CREDS_JSON") or ""
    b64 = os.getenv("GOOGLE_CREDS_JSON_B64") or ""

    raw_fallback = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or ""
    b64_fallback = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64") or ""

    if raw:
        if os.path.exists(raw):
            try:
                with open(raw, "r", encoding="utf-8") as f:
                    return json.loads(f.read())
            except Exception as e:
                raise HTTPException(500, f"Archivo de credenciales inválido en GOOGLE_CREDS_JSON: {e}")
        try:
            return json.loads(raw)
        except Exception:
            pass

    if b64:
        try:
            dec = base64.b64decode(b64).decode("utf-8")
            return json.loads(dec)
        except Exception as e:
            raise HTTPException(500, f"GOOGLE_CREDS_JSON_B64 inválido: {e}")

    if raw_fallback:
        if os.path.exists(raw_fallback):
            try:
                with open(raw_fallback, "r", encoding="utf-8") as f:
                    return json.loads(f.read())
            except Exception as e:
                raise HTTPException(500, f"Archivo inválido en GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
        try:
            return json.loads(raw_fallback)
        except Exception:
            pass

    if b64_fallback:
        try:
            dec = base64.b64decode(b64_fallback).decode("utf-8")
            return json.loads(dec)
        except Exception as e:
            raise HTTPException(500, f"GOOGLE_SERVICE_ACCOUNT_JSON_B64 inválido: {e}")

    raise HTTPException(
        500,
        "Faltan credenciales: define GOOGLE_CREDS_JSON (ruta o JSON) o GOOGLE_CREDS_JSON_B64."
    )

def get_service_account_email() -> str:
    try:
        return _load_sa_info().get("client_email", "")
    except Exception:
        return ""

def _gspread_client() -> gspread.Client:
    creds = Credentials.from_service_account_info(_load_sa_info(), scopes=SHEETS_SCOPE)
    return gspread.authorize(creds)

def extract_sheet_id(sheet_url_or_id: str) -> str:
    sheet_url_or_id = (sheet_url_or_id or "").strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", sheet_url_or_id):
        return sheet_url_or_id
    raise HTTPException(400, "URL/ID de Google Sheets inválida")

def _open_spreadsheet(sheet_id: str):
    gc = _gspread_client()
    try:
        return gc.open_by_key(sheet_id)
    except gspread.exceptions.SpreadsheetNotFound:
        # No existe o no accesible -> tratar como 403 para UX
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "La cuenta de servicio no tiene acceso al Sheet. "
                    "Compártelo con: " + get_service_account_email()
                ),
                "service_account_email": get_service_account_email(),
            },
        )
    except gspread.exceptions.APIError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "La cuenta de servicio no tiene acceso al Sheet. "
                        "Compártelo con: " + get_service_account_email()
                    ),
                    "service_account_email": get_service_account_email(),
                },
            )
        raise HTTPException(502, f"Google Sheets API error (code={code}): {e}")
    except Exception as e:
        raise HTTPException(500, f"Error abriendo spreadsheet: {e}")

def _pick_products_ws(sh: gspread.Spreadsheet) -> gspread.Worksheet:
    for name in ("Products", "Inventario", "Inventory", "Stock"):
        try:
            return sh.worksheet(name)
        except Exception:
            pass
    for ws in sh.worksheets():
        headers = [h.strip().lower() for h in (ws.row_values(1) or [])]
        score = 0
        for key in ("sku", "stock", "price", "name", "nombre", "lowthreshold"):
            if key in headers:
                score += 1
        if score >= 2:
            return ws
    return sh.sheet1

def _summarize_records(rows: list[dict]) -> dict:
    skus = set()
    low_items = 0
    for r in rows:
        sku = str(r.get("SKU", r.get("sku", ""))).strip()
        if sku:
            skus.add(sku)
        try:
            stock = int(str(r.get("Stock", r.get("stock", 0))).split()[0].replace(",", ""))
            th = int(str(r.get("LowThreshold", r.get("lowthreshold", 0))).split()[0].replace(",", ""))
            if stock <= th:
                low_items += 1
        except Exception:
            pass
    return {"sku_count": len(skus), "low_stock_items": low_items}

# --- Helpers para alertas de bajo stock (demo) ---

LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "5"))  # por defecto 5

def _first(d: dict, keys: list[str]):
    for k in keys:
        if k in d and str(d[k]).strip() != "":
            return d[k]
    return None

def _to_int(x):
    try:
        s = str(x).strip().replace(",", "")
        import re as _re
        m = _re.search(r"-?\d+", s)
        if m:
            return int(m.group(0))
        return int(float(s))
    except Exception:
        return None

def _compose_name(row: dict) -> str:
    """Construye nombre: MARCA + MODELO + Talla X (si existe)."""
    marca = str(_first(row, ["marca", "Marca", "brand", "Brand"]) or "").strip()
    modelo = str(_first(row, ["modelo", "Modelo", "model", "Model"]) or "").strip()
    talla = str(_first(row, ["talla", "Talla", "size", "Size", "numero", "número"]) or "").strip()
    parts = []
    if marca:  parts.append(marca)
    if modelo: parts.append(modelo)
    if talla:  parts.append(f"Talla {talla}")
    if not parts:
        # Fallback a nombre genérico de producto si no hay marca/modelo
        parts.append(str(_first(row, ["name", "Name", "nombre", "producto", "descripcion", "descripción"]) or "").strip())
    return " ".join([p for p in parts if p])

def _low_stock_alerts(rows: list[dict]) -> list[dict]:
    """
    Devuelve [{sku, name, stock, threshold}] cuando cantidad < LOW_STOCK_THRESHOLD.
    'name' = "MARCA MODELO Talla X" para que aparezca completo en la UI.
    """
    alerts = []
    for r in rows:
        # cantidad real (numérica). Evita usar 'stock' si es "SI/NO".
        qty = _to_int(_first(r, ["cantidad", "qty", "existencias", "unidades"]))
        if qty is None:
            qty = _to_int(_first(r, ["Stock", "stock"]))  # solo si es numérico

        if qty is None:
            continue

        if qty < LOW_STOCK_THRESHOLD:
            sku  = str(_first(r, ["SKU", "sku", "codigo", "código", "ref", "referencia"]) or "").strip()
            name = _compose_name(r)
            alerts.append({
                "sku": sku,
                "name": name,
                "stock": qty,                     # stock restante
                "threshold": LOW_STOCK_THRESHOLD  # umbral usado (5 por default)
            })
    return alerts


# ====== API: conectar y leer inventario ======
class ConnectBody(BaseModel):
    sheet: str

@app.post("/connect")
def connect_inventory(body: ConnectBody):
    sid = extract_sheet_id(body.sheet)
    sh = _open_spreadsheet(sid)
    if isinstance(sh, JSONResponse):  # 403 sin acceso
        return sh
    ws = _pick_products_ws(sh)
    rows = ws.get_all_records()
    sample = rows[:10]
    alerts = _low_stock_alerts(rows)[:50]  # demo

    summary = {
        "sheet_title": sh.title,
        "sheet_id": sid,
        "worksheet": ws.title,
        "rows_total": len(rows),
        **_summarize_records(rows),
    }
    return {
        "ok": True,
        "sheet_id": sid,
        "summary": summary,
        "low_stock_alerts": alerts,
        "sample": sample
    }

@app.get("/inv/data")
def inv_data(sid: str):
    sid = extract_sheet_id(sid)
    sh = _open_spreadsheet(sid)
    if isinstance(sh, JSONResponse):
        return sh
    ws = _pick_products_ws(sh)
    rows = ws.get_all_records()
    alerts = _low_stock_alerts(rows)[:200]  # demo

    return {
        "ok": True,
        "sheet_title": sh.title,
        "worksheet": ws.title,
        "rows_total": len(rows),
        "summary": _summarize_records(rows),
        "alerts": alerts,
        "items": rows[:200],  # muestra
    }


# ---------- diagnóstico ----------
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from urllib.parse import quote
import html

@app.get("/__routes", include_in_schema=False)
def __routes():
    return [{"path": r.path, "methods": sorted(list(getattr(r, "methods", []) or []))}
            for r in app.router.routes]

@app.get("/ping", include_in_schema=False)
def ping():
    return {"pong": True}

@app.get("/_root_test", response_class=PlainTextResponse, include_in_schema=False)
def _root_test():
    return "alive"

# ---------- Cabeceras de seguridad (mitiga advertencias) ----------
@app.middleware("http")
async def _security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline';"
    )
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp

# ---------- HOME (link inventory) ----------
# ---------- LANDING ----------
HOME_HTML = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Operaciones — Suite</title>
  <style>
    :root{--bg:#0b0d10;--card:#0f1318;--line:#1c2128;--fg:#e6eaf0;--muted:#8a94a6;--accent:#14ae5c}
    *{box-sizing:border-box} body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--fg);margin:0}
    main{max-width:960px;margin:8vh auto;padding:20px}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;text-decoration:none;color:inherit}
    .card:hover{outline:2px solid #233045}
    h1{margin:0 0 14px} .muted{color:var(--muted)}
  </style>
</head>
<body>
  <main>
    <h1>Suite de Operaciones</h1>
    <p class="muted">Elige un módulo para continuar.</p>
    <section class="grid">
      <a class="card" href="/inventory/link">
        <h2>Organizar Inventario</h2>
        <p class="muted">Conecta tu Google Sheet y gestiona stock.</p>
      </a>
      <a class="card" href="/calendar">
        <h2>Manager Calendar</h2>
        <p class="muted">Citas, confirmación y notas (demo).</p>
      </a>
      <a class="card" href="/prospects">
        <h2>Recibir CV & Prospectos</h2>
        <p class="muted">Formulario simple y carga de CV (demo).</p>
      </a>
    </section>
  </main>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTMLResponse(HOME_HTML)

DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Inventory — Link your inventory</title>
  <style>
    :root{--bg:#0b0d10;--card:#0f1318;--line:#1c2128;--fg:#e6eaf0;--muted:#8a94a6;--accent:#14ae5c}
    *{box-sizing:border-box}
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--fg);margin:0}
    main{max-width:640px;margin:10vh auto;padding:24px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 6px 16px rgba(0,0,0,.25)}
    h1{font-size:22px;margin:0 0 12px}
    p{margin:0 0 12px}
    .muted{color:var(--muted)}
    .inp{width:100%;padding:12px;border-radius:12px;border:1px solid var(--line);background:#0b0d10;color:var(--fg)}
    .actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
    .btn{display:inline-block;padding:12px 14px;border-radius:12px;border:1px solid #2e7d32;background:var(--accent);color:#fff;
         font-weight:700;cursor:pointer;text-align:center;text-decoration:none}
    .btn:disabled{opacity:.6;cursor:not-allowed}
    .btn-outline{display:inline-block;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .pill{display:inline-block;padding:4px 8px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted)}
    code{background:#0b0d10;border:1px solid var(--line);padding:3px 6px;border-radius:8px}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Link your inventory</h1>
      <p class="muted">Paste your <b>inventory link</b>. If it’s a Google Sheet, we’ll detect the ID and connect.</p>

      <label for="sheet" class="muted">Inventory URL</label>
      <input id="sheet" class="inp" type="text" placeholder="https://docs.google.com/spreadsheets/d/…">

      <div class="actions">
        <button id="actionBtn" class="btn" disabled onclick="connectNow()">Connect inventory</button>
        <button class="btn-outline" onclick="clearUrl()">Clear</button>
      </div>

      <p id="msg" class="muted" style="margin-top:8px"></p>

      <div id="savedBox" style="display:none;margin-top:10px">
        <div class="row">
          <span class="pill">URL</span><code id="showUrl"></code>
        </div>
        <div class="row" style="margin-top:8px">
          <span class="pill">Sheet ID</span><code id="showId"></code>
        </div>
      </div>
    </section>
  </main>

<script>
const $ = s => document.querySelector(s);
function showMsg(t){ $("#msg").textContent = t || ""; }

function updateButtonState(){
  const hasVal = ($("#sheet").value || "").trim().length > 0;
  $("#actionBtn").disabled = !hasVal;
}

function renderSaved(){
  const url = localStorage.getItem("inv_sheet_url") || "";
  const id  = localStorage.getItem("inv_sheet_id") || "";
  $("#savedBox").style.display = url ? "" : "none";
  $("#showUrl").textContent = url || "";
  $("#showId").textContent  = id || "";
}

async function connectNow(){
  const raw = ($("#sheet").value || "").trim();
  if(!raw){ showMsg("Enter a link first."); return; }

  const btn = $("#actionBtn");
  btn.disabled = true; const prev = btn.textContent; btn.textContent = "Connecting…";
  showMsg("");

  try{
    const res = await fetch('/connect', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ sheet: raw })
    });
    const j = await res.json().catch(()=>({}));

    if(!res.ok){
      const extra = j.service_account_email ? ` — share your Sheet with: ${j.service_account_email}` : "";
      showMsg((j.detail || "Couldn’t connect") + extra);
    }else{
      localStorage.setItem("inv_sheet_url", raw);
      localStorage.setItem("inv_sheet_id", j.sheet_id);
      renderSaved();
      showMsg("Inventory connected ✔");
      // clean route for the connected view
      window.location.href = '/inventory?sheet_id=' + encodeURIComponent(j.sheet_id);
    }
  }catch(e){
    showMsg("Network error: " + (e && e.message || e));
  }finally{
    btn.disabled = false; btn.textContent = prev;
  }
}

function clearUrl(){
  localStorage.removeItem("inv_sheet_url");
  localStorage.removeItem("inv_sheet_id");
  $("#sheet").value = "";
  updateButtonState(); renderSaved(); showMsg("Saved link removed.");
}

window.addEventListener("load", ()=>{
  const url = localStorage.getItem("inv_sheet_url") || "";
  if(url) $("#sheet").value = url;
  updateButtonState(); renderSaved();
});
window.addEventListener("input", e=>{
  if(e.target && e.target.id === "sheet") updateButtonState();
});
</script>
</body>
</html>
"""

@app.get("/inventory/link", response_class=HTMLResponse, include_in_schema=False)
def inventory_link():
    return HTMLResponse(DASHBOARD_HTML)


# Compatibilidad: /inv?sid=... -> redirige a /inventory?sheet_id=...
@app.get("/inv", include_in_schema=False)
def inv_legacy(sid: str):
    return RedirectResponse(f"/inventory?sheet_id={quote(sid)}", status_code=307)
INVENTORY_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Inventory connected — Demo</title>
<style>
  :root{
    --bg:#0b0d10;--card:#0f1318;--line:#1c2128;--fg:#e6eaf0;--muted:#8a94a6;
    --accent:#14ae5c;--warn:#ffb703;--danger:#ff6b6b
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif}
  main{max-width:1100px;margin:6vh auto;padding:24px}
  .grid{display:grid;grid-template-columns:2.2fr 1fr;gap:16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
  h1,h2{margin:0 0 12px}
  small{color:var(--muted)}
  table{width:100%;border-collapse:collapse;margin-top:10px}
  th,td{padding:8px;border-bottom:1px solid var(--line);text-align:left}
  .pill{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted)}
  .tag-demo{display:inline-block;margin-left:8px;font-size:12px;padding:2px 8px;border-radius:999px;background:#223;opacity:.8}
  .alert{border-left:4px solid var(--warn);padding-left:10px;margin:10px 0}
  .alert strong{color:#fff}
  .alert small{color:var(--muted)}
  .alert.danger{border-left-color:var(--danger)}
  .inp{width:100%;padding:10px;border-radius:10px;border:1px solid var(--line);background:#0b0d10;color:var(--fg)}
  .btn{padding:10px 12px;border-radius:10px;border:1px solid #2e7d32;background:var(--accent);color:#fff;font-weight:700;cursor:pointer}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
</style>
</head>
<body>
<main>
  <div class="grid">
    <!-- Main: table -->
    <section class="card">
      <h1>
        Inventory connected
        <span class="tag-demo">DEMO</span>
      </h1>
      <small id="meta">Loading…</small>
      <div id="wrap"></div>
    </section>

    <!-- Sidebar: alerts + email -->
    <aside class="card">
      <h2>Low stock</h2>
      <small id="alertMeta">Scanning…</small>
      <div id="alertsBox" style="margin-top:8px"></div>

      <div style="margin-top:18px">
        <h2>Notify</h2>
        <small class="pill">Visual only (saved locally)</small>
        <div class="row" style="margin-top:8px">
          <input id="notifyEmail" class="inp" type="email" placeholder="you@company.com">
          <button class="btn" onclick="saveEmail()">Save</button>
        </div>
        <small id="emailMsg" style="display:block;margin-top:6px;color:var(--muted)"></small>
      </div>
    </aside>
  </div>
</main>

<script>
(function(){
  const $ = s => document.querySelector(s);

  function renderTable(rows){
    if(!rows || !rows.length){ $("#wrap").innerHTML = '<p>No rows.</p>'; return; }
    const cols = Object.keys(rows[0]);
    let html = '<table><thead><tr>' + cols.map(c=>'<th>'+c+'</th>').join('') + '</tr></thead><tbody>';
    for(const r of rows){
      html += '<tr>' + cols.map(c=>'<td>'+(r[c] ?? '')+'</td>').join('') + '</tr>';
    }
    html += '</tbody></table>';
    $("#wrap").innerHTML = html;
  }

  function renderAlerts(alerts){
    const meta = $("#alertMeta");
    const box  = $("#alertsBox");
    if(!alerts || alerts.length===0){
      meta.textContent = 'No low-stock alerts.';
      box.innerHTML = '';
      return;
    }
    meta.textContent = alerts.length + ' alert(s).';
    const items = alerts.slice(0,50).map(a=>{
      const sev = (a.threshold !== null && a.stock !== null && a.stock <= 0) ? 'danger' : '';
      const name = (a.name || '').replace(/</g,'&lt;');
      const sku  = (a.sku  || '').replace(/</g,'&lt;');
      return `
        <div class="alert ${sev}">
          <strong>${sku || '(no SKU)'}</strong> — ${name || '(no name)'}<br>
          <small>stock: ${a.stock ?? '-'} / threshold: ${a.threshold ?? '-'}</small>
        </div>
      `;
    }).join('');
    box.innerHTML = items;
  }

  function saveEmail(){
    const v = ($("#notifyEmail").value || '').trim();
    localStorage.setItem('inv_notify_email', v);
    $("#emailMsg").textContent = v ? ('Saved: ' + v) : 'Email cleared.';
  }

  // Init
  window.addEventListener('load', async ()=>{
    $("#notifyEmail").value = localStorage.getItem('inv_notify_email') || '';

    const url = new URL(location.href);
    // accept both ?sheet_id= and ?sid= for compatibility
    const sid = url.searchParams.get('sheet_id') || url.searchParams.get('sid') ||
                (sessionStorage.getItem('sid') || localStorage.getItem('inv_sheet_id') || '');
    if(!sid){ $("#meta").textContent = 'Missing sheet_id'; return; }

    try{
      const res = await fetch('/inv/data?sid=' + encodeURIComponent(sid));
      const j = await res.json();
      if(!res.ok){ $("#meta").textContent = j.detail || 'Error'; return; }

      $("#meta").textContent =
        `${j.sheet_title} — sheet: ${j.worksheet} — rows: ${j.rows_total} — SKUs: ${j?.summary?.sku_count ?? '-'} — Low stock: ${j?.summary?.low_stock_items ?? '-'}`;

      renderTable(j.items || []);
      renderAlerts(j.alerts || []);
    }catch(e){
      $("#meta").textContent = 'Error: ' + (e && e.message || e);
    }
  });
})();
</script>
</body>
</html>
"""

# Render hace HEAD / en healthchecks -> respondemos 200
@app.head("/", include_in_schema=False)
async def ui_root_head():
    return Response(status_code=200)

# ---------- CATCH-ALL: cualquier GET vuelve al HOME ----------
@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
def catch_all(full_path: str):
    return HTMLResponse(DASHBOARD_HTML)

# ---------- Local run (opcional) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inventory:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
# ---------- Calendar (stub) ----------
CAL_HTML = r"""
<!doctype html><html lang="es"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Manager Calendar — Demo</title>
<style>
:root{--bg:#0b0d10;--card:#0f1318;--line:#1c2128;--fg:#e6eaf0;--muted:#8a94a6;--accent:#14ae5c}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif}
main{max-width:720px;margin:8vh auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
a.btn{display:inline-block;margin-bottom:12px;padding:8px 12px;border:1px solid var(--line);border-radius:10px;color:var(--fg);text-decoration:none}
input{width:100%;padding:10px;border-radius:10px;border:1px solid var(--line);background:#0b0d10;color:var(--fg)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{padding:10px 12px;border-radius:10px;border:1px solid #2e7d32;background:var(--accent);color:#04160b;font-weight:800;cursor:pointer}
</style></head><body>
<main>
  <a class="btn" href="/">← Volver</a>
  <section class="card">
    <h2>Nueva cita (demo)</h2>
    <div class="row">
      <input id="name" placeholder="Nombre"/>
      <input id="date" type="date"/>
    </div>
    <div class="row" style="margin-top:10px">
      <input id="time" type="time"/>
      <input id="notes" placeholder="Notas"/>
    </div>
    <div style="margin-top:10px"><button onclick="alert('Guardado (demo)')">Guardar</button></div>
  </section>
</main></body></html>
"""

@app.get("/calendar", response_class=HTMLResponse, include_in_schema=False)
def calendar_page():
    return HTMLResponse(CAL_HTML)

# ---------- Prospects (stub) ----------
PROS_HTML = r"""
<!doctype html><html lang="es"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Prospectos & CV — Demo</title>
<style>
:root{--bg:#0b0d10;--card:#0f1318;--line:#1c2128;--fg:#e6eaf0;--muted:#8a94a6;--accent:#14ae5c}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif}
main{max-width:720px;margin:8vh auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
a.btn{display:inline-block;margin-bottom:12px;padding:8px 12px;border:1px solid var(--line);border-radius:10px;color:var(--fg);text-decoration:none}
input,textarea{width:100%;padding:10px;border-radius:10px;border:1px solid var(--line);background:#0b0d10;color:var(--fg)}
label{display:block;margin:10px 0 6px;color:var(--muted)}
button{padding:10px 12px;border-radius:10px;border:1px solid #2e7d32;background:var(--accent);color:#04160b;font-weight:800;cursor:pointer}
</style></head><body>
<main>
  <a class="btn" href="/">← Volver</a>
  <section class="card">
    <h2>Recibir CV & Prospectos (demo)</h2>
    <label>Nombre</label><input placeholder="Nombre completo"/>
    <label>Email</label><input type="email" placeholder="correo@ejemplo.com"/>
    <label>Puesto de interés</label><input placeholder="Cargo"/>
    <label>CV</label><input type="file" disabled/>
    <div style="margin-top:10px"><button onclick="alert('Enviado (demo)')">Enviar</button></div>
    <p class="muted" style="margin-top:8px">*Luego conectamos esto a backend para guardar archivos y datos.</p>
  </section>
</main></body></html>
"""

@app.get("/prospects", response_class=HTMLResponse, include_in_schema=False)
def prospects_page():
    return HTMLResponse(PROS_HTML)
# Al final del archivo, después de /calendar y /prospects
@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
def catch_all(full_path: str):
    return HTMLResponse(HOME_HTML)
