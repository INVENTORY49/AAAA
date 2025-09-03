# inventory.py — mínimal FastAPI para Render

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import os, time, logging

APP_NAME = "Inventory demo"

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# -------- logging + diagnóstico --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("inventory")

@app.on_event("startup")
async def _startup():
    log.info("Startup con %d rutas", len(app.router.routes))
    for r in app.router.routes:
        try:
            methods = ",".join(sorted(getattr(r, "methods", []) or []))
        except Exception:
            methods = ""
        log.info("   -> %s %s", methods, r.path)

@app.middleware("http")
async def _log_req(request: Request, call_next):
    t0 = time.time()
    try:
        resp = await call_next(request)
    except Exception:
        log.exception("Unhandled %s %s", request.method, request.url.path)
        raise
    dt = (time.time() - t0) * 1000
    log.info("%s %s -> %s (%.1f ms)",
             request.method, request.url.path,
             getattr(resp, "status_code", "?"), dt)
    return resp

@app.exception_handler(StarletteHTTPException)
async def _http_exc(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        paths = [r.path for r in app.router.routes]
        log.warning("404 en %s. Rutas conocidas=%s", request.url.path, paths)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

@app.get("/__routes", include_in_schema=False)
def __routes():
    return [{"path": r.path,
             "methods": sorted(list(getattr(r, "methods", []) or []))}
            for r in app.router.routes]

@app.get("/ping", include_in_schema=False)
def ping():
    return {"pong": True}

# -------- UI básica --------
ROOT_HTML = """<!doctype html><meta charset="utf-8">
<title>OK</title><body style="font-family:system-ui;background:#0b0d10;color:#e6eaf0">
<h1>Servicio en línea ✅</h1>
<p>Ir a <a href="/ui">/ui</a>.</p></body>"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return HTMLResponse(ROOT_HTML)

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse("<h1>UI</h1><p>Si ves esto, el enrutamiento funciona.</p>")

# Para correr localmente si quieres
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inventory:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
