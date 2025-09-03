# inventory.py — FastAPI mínimo con diagnóstico y catch-all
import os, sys, time, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

APP_NAME = "Inventory demo (catch-all)"

# --- huella en logs al importar el módulo
print(">>> [inventory] importado desde:", __file__, "  cwd:", os.getcwd(), "  sys.path[0]:", sys.path[0], flush=True)

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --- logging
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

# --- diagnóstico
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

# --- UI mínima
ROOT_HTML = """<!doctype html><meta charset="utf-8">
<title>OK</title><body style="font-family:system-ui;background:#0b0d10;color:#e6eaf0">
<h1>Servicio en línea ✅</h1>
<p>Ruta raíz funcionando. Ver <a href="/__routes">/__routes</a> y <a href="/ping">/ping</a>.</p>
</body>"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return HTMLResponse(ROOT_HTML)

# --- CATCH-ALL: cualquier GET que no matchee otra ruta vuelve al root (sin 404)
@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
def catch_all(full_path: str):
    # OJO: las rutas reales registradas arriba tienen prioridad.
    return HTMLResponse(ROOT_HTML)

# Local run (opcional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inventory:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
