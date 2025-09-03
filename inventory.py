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

# ---------- HOME (HTML embebido) ----------
DASHBOARD_HTML = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Inventory — Inicio</title>
  <style>
    :root{--bg:#0b0d10;--card:#0f1318;--line:#1c2128;--fg:#e6eaf0;--muted:#8a94a6;--accent:#14ae5c}
    *{box-sizing:border-box}
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--fg);margin:0}
    main{max-width:520px;margin:10vh auto;padding:24px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 6px 16px rgba(0,0,0,.25)}
    h1{font-size:20px;margin:0 0 16px}
    p{margin:0 0 12px}
    .btn{display:block;width:100%;padding:12px;border-radius:12px;border:1px solid #2e7d32;background:var(--accent);color:#fff;
         font-weight:700;cursor:pointer;text-align:center;text-decoration:none;margin-top:10px}
    .row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .row a{flex:1}
    .muted{color:var(--muted)}
    pre{white-space:pre-wrap}
    a:link,a:visited{color:#9fd8ff}
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>Inventory — Inicio</h1>
      <p class="muted">HTML embebido en el mismo archivo. Modifica aquí y se refleja al instante.</p>

      <a class="btn" href="/__routes" target="_blank">Ver rutas</a>

      <div class="row">
        <a class="btn" href="/docs" target="_blank">Docs</a>
        <a class="btn" href="/ping" target="_blank">Ping</a>
      </div>

      <div style="margin-top:14px">
        <small class="muted">Estado:</small>
        <pre id="out" class="muted">Cargando…</pre>
      </div>
    </div>
  </main>

<script>
(async function(){
  try{
    const [pingRes, routesRes] = await Promise.all([
      fetch('/ping').then(r => r.json()).catch(()=>({})),
      fetch('/__routes').then(r => r.json()).catch(()=>[])
    ]);
    document.getElementById('out').textContent =
      JSON.stringify({ ping: pingRes, routes_count: (routesRes||[]).length }, null, 2);
  }catch(e){
    document.getElementById('out').textContent = 'Error: ' + (e && e.message || e);
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
