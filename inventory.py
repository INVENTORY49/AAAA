# inventory.py — FastAPI mínimo con diagnóstico y home embebido
import os, sys, time, logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

APP_NAME = "Inventory demo (catch-all)"

# Huella al importar
print(">>> [inventory] importado desde:", __file__, "  cwd:", os.getcwd(), "  sys.path[0]:", sys.path[0], flush=True)

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

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
    # Preferidas en Render:
    raw = os.getenv("GOOGLE_CREDS_JSON") or ""
    b64 = os.getenv("GOOGLE_CREDS_JSON_B64") or ""

    # Compat (por si ya las tienes puestas):
    raw_fallback = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or ""
    b64_fallback = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64") or ""

    # 1) GOOGLE_CREDS_JSON (ruta o JSON)
    if raw:
        if os.path.exists(raw):  # archivo
            try:
                with open(raw, "r", encoding="utf-8") as f:
                    return json.loads(f.read())
            except Exception as e:
                raise HTTPException(500, f"Archivo de credenciales inválido en GOOGLE_CREDS_JSON: {e}")
        # JSON plano
        try:
            return json.loads(raw)
        except Exception:
            pass

    # 2) GOOGLE_CREDS_JSON_B64 (Base64)
    if b64:
        try:
            dec = base64.b64decode(b64).decode("utf-8")
            return json.loads(dec)
        except Exception as e:
            raise HTTPException(500, f"GOOGLE_CREDS_JSON_B64 inválido: {e}")

    # 3) Compat: GOOGLE_SERVICE_ACCOUNT_JSON (ruta o JSON)
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

    # 4) Compat: GOOGLE_SERVICE_ACCOUNT_JSON_B64 (Base64)
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
        # 403 con email de la SA para mostrar en el front
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

def _pick_products_ws(sh: gspread.Spreadsheet) -> gspread.Worksheet:
    # 1) Nombres comunes
    for name in ("Products", "Inventario", "Inventory", "Stock"):
        try:
            return sh.worksheet(name)
        except Exception:
            pass
    # 2) Heurística por headers
    for ws in sh.worksheets():
        headers = [h.strip().lower() for h in (ws.row_values(1) or [])]
        score = 0
        for key in ("sku", "stock", "price", "name", "nombre", "lowthreshold"):
            if key in headers:
                score += 1
        if score >= 2:
            return ws
    # 3) Fallback: la primera
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
    summary = {
        "sheet_title": sh.title,
        "sheet_id": sid,
        "worksheet": ws.title,
        "rows_total": len(rows),
        **_summarize_records(rows),
    }
    return {"ok": True, "sheet_id": sid, "summary": summary, "sample": sample}

@app.get("/inv/data")
def inv_data(sid: str):
    sid = extract_sheet_id(sid)
    sh = _open_spreadsheet(sid)
    if isinstance(sh, JSONResponse):
        return sh
    ws = _pick_products_ws(sh)
    rows = ws.get_all_records()
    return {
        "ok": True,
        "sheet_title": sh.title,
        "worksheet": ws.title,
        "rows_total": len(rows),
        "summary": _summarize_records(rows),
        "items": rows[:200],  # muestra
    }





# ---------- diagnóstico ----------
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

DASHBOARD_HTML = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Inventory — Vincular inventario</title>
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
      <h1>Vincula aquí tu inventario</h1>
      <p class="muted">Pega el <b>link del inventario</b>. Si es Google Sheets, detectaremos el ID y lo conectaremos.</p>

      <label for="sheet" class="muted">URL del inventario</label>
      <input id="sheet" class="inp" type="text" placeholder="https://docs.google.com/spreadsheets/d/…">

      <div class="actions">
        <button id="actionBtn" class="btn" disabled onclick="connectNow()">Conectar con inventario</button>
        <button class="btn-outline" onclick="clearUrl()">Quitar</button>
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

function extractSheetId(v){
  if(!v) return null;
  const m = v.match(/\/spreadsheets\/d\/([a-zA-Z0-9-_]+)/);
  if(m) return m[1];
  if(/^[a-zA-Z0-9-_]{20,}$/.test(v)) return v;
  return null;
}

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
  if(!raw){ showMsg("Escribe un enlace primero."); return; }

  const btn = $("#actionBtn");
  btn.disabled = true; const prev = btn.textContent; btn.textContent = "Conectando…";
  showMsg("");

  try{
    const res = await fetch('/connect', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ sheet: raw })
    });
    const j = await res.json().catch(()=>({}));

    if(!res.ok){
      const extra = j.service_account_email ? ` — comparte tu Sheet con: ${j.service_account_email}` : "";
      showMsg((j.detail || "No se pudo conectar") + extra);
    }else{
      // guardamos
      localStorage.setItem("inv_sheet_url", raw);
      localStorage.setItem("inv_sheet_id", j.sheet_id);
      renderSaved();
      showMsg("Inventario conectado ✔");
      // abre la interfaz de inventario en nueva pestaña
      window.open('/inv?sid=' + encodeURIComponent(j.sheet_id), '_blank');
    }
  }catch(e){
    showMsg("Error de red: " + (e && e.message || e));
  }finally{
    btn.disabled = false; btn.textContent = prev;
  }
}

function clearUrl(){
  localStorage.removeItem("inv_sheet_url");
  localStorage.removeItem("inv_sheet_id");
  $("#sheet").value = "";
  updateButtonState(); renderSaved(); showMsg("Se eliminó el vínculo guardado.");
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
@app.get("/inv", response_class=HTMLResponse)
def inv_page():
    return HTMLResponse(INVENTORY_HTML)

INVENTORY_HTML = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Inventario conectado</title>
<style>
  :root{--bg:#0b0d10;--card:#0f1318;--line:#1c2128;--fg:#e6eaf0;--muted:#8a94a6}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif}
  main{max-width:980px;margin:6vh auto;padding:24px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
  h1{margin:0 0 12px}
  table{width:100%;border-collapse:collapse;margin-top:10px}
  th,td{padding:8px;border-bottom:1px solid var(--line);text-align:left}
  small{color:var(--muted)}
</style>
</head>
<body>
<main>
  <div class="card">
    <h1>Inventario conectado</h1>
    <small id="meta">Cargando…</small>
    <div id="wrap"></div>
  </div>
</main>
<script>
(async function(){
  const url = new URL(location.href);
  const sid = url.searchParams.get('sid') || (sessionStorage.getItem('sid') || localStorage.getItem('inv_sheet_id') || '');
  if(!sid){ document.getElementById('meta').textContent = 'Falta sid'; return; }

  try{
    const res = await fetch('/inv/data?sid=' + encodeURIComponent(sid));
    const j = await res.json();
    if(!res.ok){ document.getElementById('meta').textContent = j.detail || 'Error'; return; }

    document.getElementById('meta').textContent =
      `${j.sheet_title} — hoja: ${j.worksheet} — filas: ${j.rows_total} — SKUs: ${j.summary?.sku_count ?? '-'} — Low stock: ${j.summary?.low_stock_items ?? '-'}`;

    const rows = j.items || [];
    if(!rows.length){ document.getElementById('wrap').innerHTML = '<p>No hay filas.</p>'; return; }

    const cols = Object.keys(rows[0]);
    let html = '<table><thead><tr>' + cols.map(c=>'<th>'+c+'</th>').join('') + '</tr></thead><tbody>';
    for(const r of rows){
      html += '<tr>' + cols.map(c=>'<td>'+(r[c] ?? '')+'</td>').join('') + '</tr>';
    }
    html += '</tbody></table>';
    document.getElementById('wrap').innerHTML = html;
  }catch(e){
    document.getElementById('meta').textContent = 'Error: ' + (e && e.message || e);
  }
})();
</script>
</body>
</html>
"""



@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_root():
    return HTMLResponse(DASHBOARD_HTML)

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
