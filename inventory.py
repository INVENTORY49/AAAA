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
    main{max-width:640px;margin:8vh auto;padding:24px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 6px 16px rgba(0,0,0,.25);margin-bottom:16px}
    h1{font-size:22px;margin:0 0 12px}
    h2{font-size:18px;margin:0 0 8px}
    p{margin:0 0 12px}
    .muted{color:var(--muted)}
    a:link,a:visited{color:#9fd8ff}

    .btn{display:inline-block;padding:12px 14px;border-radius:12px;border:1px solid #2e7d32;background:var(--accent);color:#fff;
         font-weight:700;cursor:pointer;text-align:center;text-decoration:none}
    .btn:disabled{opacity:.6;cursor:not-allowed}
    .btn-outline{display:inline-block;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}

    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .actions{display:flex;gap:8px;margin-top:10px}
    .inp{width:100%;padding:12px;border-radius:12px;border:1px solid var(--line);background:#0b0d10;color:var(--fg)}
    code{background:#0b0d10;border:1px solid var(--line);padding:3px 6px;border-radius:8px}
    .pill{display:inline-block;padding:4px 8px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted)}
    pre{white-space:pre-wrap}
  </style>
</head>
<body>
  <main>

    <!-- Vincular inventario -->
    <section class="card">
      <h1>Vincula tu inventario</h1>
      <p class="muted">Pega aquí el <b>link de tu inventario</b> (puede ser una URL cualquiera; si es Google Sheets detectamos el ID). Por ahora solo lo guardamos en tu navegador para usarlo luego.</p>

      <label for="sheet" class="muted">URL del inventario</label>
      <input id="sheet" class="inp" type="text" placeholder="https://docs.google.com/spreadsheets/d/…">

      <div class="actions">
        <button class="btn" onclick="saveUrl()">Guardar vínculo</button>
        <button class="btn-outline" onclick="clearUrl()">Quitar</button>
      </div>

      <p id="msg" class="muted" style="margin-top:8px"></p>

      <div id="savedBox" style="display:none;margin-top:10px">
        <h2>Guardado</h2>
        <div class="row">
          <span class="pill">URL</span>
          <code id="showUrl"></code>
          <button class="btn-outline" onclick="copyText('#showUrl')">Copiar</button>
        </div>
        <div class="row" style="margin-top:8px">
          <span class="pill">Sheet ID</span>
          <code id="showId"></code>
          <button class="btn-outline" onclick="copyText('#showId')">Copiar</button>
        </div>
      </div>
    </section>

    <!-- Utilidades / estado -->
    <section class="card">
      <h2>Utilidades</h2>
      <div class="row">
        <a class="btn" href="/__routes" target="_blank">Ver rutas</a>
        <a class="btn" href="/docs" target="_blank">Docs</a>
        <a class="btn" href="/ping" target="_blank">Ping</a>
      </div>

      <div style="margin-top:14px">
        <small class="muted">Estado:</small>
        <pre id="out" class="muted">Cargando…</pre>
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
  if(/^[a-zA-Z0-9-_]{20,}$/.test(v)) return v; // parece ID cruda
  return null;
}

function renderSaved(){
  const url = localStorage.getItem("inv_sheet_url") || "";
  const id  = localStorage.getItem("inv_sheet_id") || "";
  if(url){
    $("#savedBox").style.display = "";
    $("#showUrl").textContent = url;
    $("#showId").textContent  = id || "(ID no detectada)";
  }else{
    $("#savedBox").style.display = "none";
  }
}

function saveUrl(){
  const raw = ($("#sheet").value || "").trim();
  if(!raw){ showMsg("Escribe un enlace primero."); return; }
  localStorage.setItem("inv_sheet_url", raw);
  const id = extractSheetId(raw);
  if(id) localStorage.setItem("inv_sheet_id", id);
  else   localStorage.removeItem("inv_sheet_id");
  renderSaved();
  showMsg("¡Listo! Enlace guardado en este navegador.");
}

function clearUrl(){
  localStorage.removeItem("inv_sheet_url");
  localStorage.removeItem("inv_sheet_id");
  $("#sheet").value = "";
  renderSaved();
  showMsg("Se eliminó el vínculo guardado.");
}

async function copyText(sel){
  const el = $(sel);
  const txt = (el && el.textContent || "").trim();
  if(!txt) return;
  try { await navigator.clipboard.writeText(txt); showMsg("Copiado al portapapeles."); }
  catch(_){ showMsg("No se pudo copiar. Selecciona y copia manualmente."); }
}

window.addEventListener("load", async ()=>{
  // precarga desde localStorage
  const url = localStorage.getItem("inv_sheet_url") || "";
  if(url) $("#sheet").value = url;
  renderSaved();

  // estado simple (ping + conteo de rutas)
  try{
    const [pingRes, routesRes] = await Promise.all([
      fetch('/ping').then(r => r.json()).catch(()=>({})),
      fetch('/__routes').then(r => r.json()).catch(()=>[])
    ]);
    $("#out").textContent = JSON.stringify({ ping: pingRes, routes_count: (routesRes||[]).length }, null, 2);
  }catch(e){
    $("#out").textContent = 'Error: ' + (e && e.message || e);
  }
});
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
