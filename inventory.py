# app.py
# ---------------------------------------------
# AI Inventory & Order Manager — DEMO
# FastAPI service that links to a client's Google Sheets as the data source
# Uses a Google Service Account (share the Sheet with the SA email) 
# ---------------------------------------------

import os
import re
import json
import random
import string
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr, Field

import gspread
from google.oauth2.service_account import Credentials

APP_NAME = "AI Inventory & Order Manager — DEMO"
DATA_FILE = "stores.json"  # simple persistence (demo)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ----------------------------
# Helpers: Service Account & GSpread
# ----------------------------

def _load_sa_info() -> dict:
    sa_env = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_env:
        raise RuntimeError(
            "Missing GOOGLE_SERVICE_ACCOUNT_JSON env var. Paste your service account JSON content there."
        )
    try:
        info = json.loads(sa_env)
    except json.JSONDecodeError as e:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON") from e
    return info


def get_gspread_client() -> gspread.Client:
    info = _load_sa_info()
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_service_account_email() -> str:
    info = _load_sa_info()
    return info.get("client_email", "")


# ----------------------------
# Simple persistence (demo): map store_token -> sheet_id
# ----------------------------

def _load_store_map() -> Dict[str, str]:
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_store_map(data: Dict[str, str]) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------
# Sheets Helpers
# ----------------------------
SHEET_PRODUCTS = "Products"
SHEET_ORDERS = "Orders"
SHEET_SHIPMENTS = "Shipments"

PRODUCT_HEADERS = ["SKU", "Name", "Price", "Stock", "LowThreshold"]
ORDER_HEADERS = ["OrderID", "Date", "Customer", "Email", "ItemsJSON", "Total", "Status"]
SHIP_HEADERS = ["OrderID", "Carrier", "Tracking", "ETA", "ShipStatus"]


def extract_sheet_id(sheet_url_or_id: str) -> str:
    # Accept both full URL or raw ID
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url_or_id)
    if m:
        return m.group(1)
    # If it looks like an ID
    if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", sheet_url_or_id):
        return sheet_url_or_id
    raise HTTPException(400, "Invalid Google Sheets URL or ID")


def ensure_template(spreadsheet: gspread.Spreadsheet) -> None:
    existing = {ws.title for ws in spreadsheet.worksheets()}

    if SHEET_PRODUCTS not in existing:
        ws = spreadsheet.add_worksheet(title=SHEET_PRODUCTS, rows=100, cols=10)
        ws.append_row(PRODUCT_HEADERS)
    if SHEET_ORDERS not in existing:
        ws = spreadsheet.add_worksheet(title=SHEET_ORDERS, rows=1000, cols=20)
        ws.append_row(ORDER_HEADERS)
    if SHEET_SHIPMENTS not in existing:
        ws = spreadsheet.add_worksheet(title=SHEET_SHIPMENTS, rows=500, cols=20)
        ws.append_row(SHIP_HEADERS)


def open_store_spreadsheet(store_token: str) -> gspread.Spreadsheet:
    store_map = _load_store_map()
    sheet_id = store_map.get(store_token)
    if not sheet_id:
        raise HTTPException(404, "Unknown store token. Link the store first.")
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(sheet_id)
    except gspread.exceptions.SpreadsheetNotFound:
        raise HTTPException(
            403,
            "The service account does not have access to this Sheet. Please Share it with: "
            + get_service_account_email(),
        )
    ensure_template(sh)
    return sh


def get_ws(spreadsheet: gspread.Spreadsheet, title: str) -> gspread.Worksheet:
    return spreadsheet.worksheet(title)


def get_all_records(ws: gspread.Worksheet) -> List[Dict[str, Any]]:
    return ws.get_all_records()  # relies on header row being present


def generate_order_id() -> str:
    salt = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{salt}"


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------- Models ---------
class LinkBody(BaseModel):
    sheet: str = Field(..., description="Google Sheet URL or ID")
    store_name: Optional[str] = Field(None, description="A friendly name for the store (optional)")


class OrderItem(BaseModel):
    sku: str
    qty: int = Field(gt=0)


class CreateOrderBody(BaseModel):
    store: str
    customer: str
    email: Optional[EmailStr] = None
    items: List[OrderItem]


class ShipmentBody(BaseModel):
    store: str
    order_id: str
    carrier: str
    tracking: str
    eta: Optional[str] = None  # free text ETA
    ship_status: str = "IN_TRANSIT"


# --------- API Routes ---------
@app.get("/service-account")
def service_account_info():
    return {"service_account_email": get_service_account_email()}


@app.post("/link")
def link_store(body: LinkBody):
    sheet_id = extract_sheet_id(body.sheet)
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(sheet_id)
    except gspread.exceptions.SpreadsheetNotFound:
        raise HTTPException(
            403,
            "The service account does not have access to this Sheet. Share it with: "
            + get_service_account_email(),
        )

    # Ensure the 3 worksheets exist with headers
    ensure_template(sh)

    # Create a store token and persist
    token = generate_order_id().replace("ORD-", "STORE-")
    store_map = _load_store_map()
    store_map[token] = sheet_id
    _save_store_map(store_map)

    return {
        "ok": True,
        "store": token,
        "sheet_id": sheet_id,
        "message": "Linked successfully. Keep this token to authenticate your requests.",
    }


@app.get("/inventory")
def list_inventory(store: str):
    sh = open_store_spreadsheet(store)
    ws = get_ws(sh, SHEET_PRODUCTS)
    data = get_all_records(ws)
    return {"items": data}


@app.post("/order")
def create_order(body: CreateOrderBody):
    sh = open_store_spreadsheet(body.store)
    ws_products = get_ws(sh, SHEET_PRODUCTS)
    ws_orders = get_ws(sh, SHEET_ORDERS)

    products = ws_products.get_all_records()  # list of dicts
    if not products:
        raise HTTPException(400, "Products sheet is empty. Add products first.")

    # Build index: SKU -> (row_index, record)
    header_row = ws_products.row_values(1)
    try:
        sku_col = header_row.index("SKU") + 1
        stock_col = header_row.index("Stock") + 1
        price_col = header_row.index("Price") + 1
        low_col = header_row.index("LowThreshold") + 1
    except ValueError:
        raise HTTPException(400, f"Products sheet must have headers: {PRODUCT_HEADERS}")

    sku_to_row: Dict[str, int] = {}
    sku_to_record: Dict[str, Dict[str, Any]] = {}

    # records start at row 2
    for i, rec in enumerate(products, start=2):
        sku = str(rec.get("SKU", "")).strip()
        if sku:
            sku_to_row[sku] = i
            sku_to_record[sku] = rec

    # Validate & compute
    total = 0.0
    low_stock_alerts = []
    stock_updates: Dict[int, int] = {}  # row -> new_stock

    normalized_items = []
    for it in body.items:
        sku = it.sku.strip()
        rec = sku_to_record.get(sku)
        if not rec:
            raise HTTPException(400, f"SKU not found: {sku}")
        try:
            stock = int(rec.get("Stock", 0))
            price = float(rec.get("Price", 0))
            low_th = int(rec.get("LowThreshold", 0))
        except Exception:
            raise HTTPException(400, f"Invalid numeric values in product row for SKU {sku}")

        if it.qty > stock:
            raise HTTPException(400, f"Insufficient stock for {sku}. Available: {stock}")

        new_stock = stock - it.qty
        row = sku_to_row[sku]
        stock_updates[row] = new_stock
        total += price * it.qty
        if new_stock <= low_th:
            low_stock_alerts.append({"sku": sku, "new_stock": new_stock, "threshold": low_th})
        normalized_items.append({"sku": sku, "qty": it.qty, "price": price, "subtotal": price * it.qty})

    # Write order row
    order_id = generate_order_id()
    now_iso = datetime.utcnow().isoformat()
    order_row = [
        order_id,
        now_iso,
        body.customer,
        body.email or "",
        json.dumps(normalized_items, ensure_ascii=False),
        f"{total:.2f}",
        "CONFIRMED",
    ]
    ws_orders.append_row(order_row)

    # Apply stock updates
    for row, new_stock in stock_updates.items():
        ws_products.update_cell(row, stock_col, str(new_stock))

    return {
        "ok": True,
        "order_id": order_id,
        "total": round(total, 2),
        "low_stock_alerts": low_stock_alerts,
    }


@app.get("/orders/{order_id}")
def get_order(order_id: str, store: str):
    sh = open_store_spreadsheet(store)
    ws_orders = get_ws(sh, SHEET_ORDERS)
    rows = ws_orders.get_all_records()
    for rec in rows:
        if str(rec.get("OrderID")) == order_id:
            # items JSON back to list
            try:
                items = json.loads(rec.get("ItemsJSON", "[]"))
            except Exception:
                items = []
            rec["Items"] = items
            return rec
    raise HTTPException(404, "Order not found")


@app.post("/shipment")
def create_or_update_shipment(body: ShipmentBody):
    sh = open_store_spreadsheet(body.store)
    ws_ship = get_ws(sh, SHEET_SHIPMENTS)

    # Just append new shipment record for demo
    ws_ship.append_row([
        body.order_id,
        body.carrier,
        body.tracking,
        body.eta or "",
        body.ship_status,
    ])
    return {"ok": True}


# ----------------------------
# Minimal HTML UI (very demo)
# ----------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    sa_email = get_service_account_email()
    html = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__APP_NAME__</title>
<style>
  :root{--bg:#0b0d10;--card:#101317;--muted:#8a94a6;--fg:#e6eaf0;--accent:#14ae5c;--line:#1f2630}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--fg)}
  main{max-width:980px;margin:5vh auto;padding:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:14px}
  input,button,textarea{font-size:15px}
  input[type=text],input[type=email]{width:100%;padding:10px;border-radius:10px;border:1px solid var(--line);background:#0b0d10;color:var(--fg)}
  button{padding:10px 14px;border-radius:10px;border:1px solid #2e7d32;background:var(--accent);color:#fff;font-weight:700;cursor:pointer}
  table{width:100%;border-collapse:collapse}
  th,td{padding:8px;border-bottom:1px solid var(--line);text-align:left}
  th{opacity:.8}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .muted{color:var(--muted)}
  .pill{display:inline-block;padding:4px 8px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted)}
</style>
</head>
<body>
<main>
  <h1 style="margin-bottom:10px">__APP_NAME__</h1>
  <div class="muted" style="margin-bottom:18px">
    Comparte tu Google Sheets con <span class="pill">__SA_EMAIL__</span> y pega el enlace abajo. Quedarás vinculado con un token de tienda.
  </div>

  <section class="card">
    <h3>1) Vincular tu tienda (Google Sheets)</h3>
    <div class="row">
      <div>
        <label>Enlace o ID de Google Sheets</label>
        <input id="sheet" placeholder="https://docs.google.com/spreadsheets/d/…" type="text">
      </div>
      <div>
        <label>Nombre de tienda (opcional)</label>
        <input id="store_name" placeholder="Mi Tienda" type="text">
      </div>
    </div>
    <div style="margin-top:10px"><button onclick="linkStore()">Vincular</button></div>
    <div id="link_msg" class="muted" style="margin-top:8px"></div>
  </section>

  <section class="card">
    <h3>2) Inventario</h3>
    <div class="row">
      <div><label>Store Token</label><input id="store_token" placeholder="STORE-…" type="text"></div>
      <div style="display:flex;align-items:end"><button onclick="loadInventory()">Cargar inventario</button></div>
    </div>
    <div id="inv_wrap" style="margin-top:12px"></div>
  </section>

  <section class="card">
    <h3>3) Crear pedido</h3>
    <div class="row">
      <div><label>Cliente</label><input id="cust" type="text" placeholder="Juan Perez"></div>
      <div><label>Email (opcional)</label><input id="email" type="email" placeholder="juan@example.com"></div>
    </div>
    <div style="margin-top:10px">
      <table id="items_tbl">
        <thead><tr><th>SKU</th><th>Cantidad</th><th></th></tr></thead>
        <tbody id="items_body"></tbody>
      </table>
      <button style="margin-top:6px" onclick="addItemRow()">+ Agregar ítem</button>
    </div>
    <div style="margin-top:10px"><button onclick="submitOrder()">Confirmar pedido</button></div>
    <div id="order_msg" class="muted" style="margin-top:8px"></div>
  </section>

  <section class="card">
    <h3>4) Rastrear pedido</h3>
    <div class="row">
      <div><label>Order ID</label><input id="oid" type="text" placeholder="ORD-…"></div>
      <div style="display:flex;align-items:end"><button onclick="trackOrder()">Ver estado</button></div>
    </div>
    <pre id="track_out" class="muted" style="margin-top:10px;white-space:pre-wrap"></pre>
  </section>

  <section class="card">
    <h3>5) Registrar envío</h3>
    <div class="row">
      <div><label>Order ID</label><input id="ship_oid" type="text" placeholder="ORD-…"></div>
      <div><label>Transportadora</label><input id="carrier" type="text" placeholder="DHL"></div>
    </div>
    <div class="row" style="margin-top:10px">
      <div><label>Tracking</label><input id="tracking" type="text" placeholder="123ABC"></div>
      <div><label>ETA (opcional)</label><input id="eta" type="text" placeholder="2025-09-10"></div>
    </div>
    <div style="margin-top:10px"><button onclick="saveShipment()">Guardar envío</button></div>
    <div id="ship_msg" class="muted" style="margin-top:8px"></div>
  </section>

</main>
<script>
const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

function setToken(t){
  document.querySelector('#store_token').value = t;
  localStorage.setItem('store_token', t);
}
function getToken(){
  return document.querySelector('#store_token').value || localStorage.getItem('store_token') || '';
}
window.addEventListener('load', ()=>{
  const t = localStorage.getItem('store_token');
  if(t) document.querySelector('#store_token').value = t;
  addItemRow();
});

async function linkStore(){
  const sheet = document.querySelector('#sheet').value.trim();
  const store_name = document.querySelector('#store_name').value.trim();
  document.querySelector('#link_msg').textContent = 'Vinculando…';
  try{
    const r = await fetch('/link', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({sheet, store_name})});
    const j = await r.json();
    if(!r.ok) throw new Error(j.detail || 'Error');
    setToken(j.store);
    document.querySelector('#link_msg').textContent = 'Listo. Token: ' + j.store;
  }catch(e){
    document.querySelector('#link_msg').textContent = 'Error: ' + e.message;
  }
}

async function loadInventory(){
  const store = getToken(); if(!store){ alert('Falta Store Token'); return; }
  const r = await fetch('/inventory?store=' + encodeURIComponent(store));
  const j = await r.json();
  if(!r.ok){ document.querySelector('#inv_wrap').textContent = j.detail || 'Error'; return; }
  const rows = j.items || [];
  if(rows.length===0){ document.querySelector('#inv_wrap').textContent = 'No hay productos'; return; }
  let html = '<table><thead><tr>';
  const cols = Object.keys(rows[0]);
  for(const c of cols) html += '<th>' + c + '</th>';
  html += '</tr></thead><tbody>';
  for(const rec of rows){
    html += '<tr>';
    for(const c of cols){ html += '<td>' + (rec[c] ?? '') + '</td>'; }
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.querySelector('#inv_wrap').innerHTML = html;
}

function addItemRow(){
  const tr = document.createElement('tr');
  tr.innerHTML = '<td><input placeholder="SKU"></td><td><input type="number" min="1" value="1"></td><td><button onclick="this.closest(\\'tr\\').remove()">Eliminar</button></td>';
  document.querySelector('#items_body').appendChild(tr);
}

async function submitOrder(){
  const store = getToken(); if(!store){ alert('Falta Store Token'); return; }
  const customer = document.querySelector('#cust').value.trim(); if(!customer){ alert('Falta nombre del cliente'); return; }
  const email = document.querySelector('#email').value.trim();
  const items = [];
  for(const tr of document.querySelectorAll('#items_body tr')){
    const sku = tr.querySelector('td:nth-child(1) input').value.trim();
    const qty = parseInt(tr.querySelector('td:nth-child(2) input').value,10) || 0;
    if(sku && qty>0) items.push({sku, qty});
  }
  if(items.length===0){ alert('Agrega al menos un ítem'); return; }
  document.querySelector('#order_msg').textContent = 'Creando pedido…';
  try{
    const r = await fetch('/order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({store, customer, email, items})});
    const j = await r.json();
    if(!r.ok) throw new Error(j.detail || 'Error');
    const alerts = (j.low_stock_alerts && j.low_stock_alerts.length)
      ? ' — ALERTAS: ' + j.low_stock_alerts.map(a => a.sku + ':' + a.new_stock).join(', ')
      : '';
    document.querySelector('#order_msg').textContent = '✔ Pedido confirmado. ID: ' + j.order_id + ' — Total: $' + j.total + alerts;
  }catch(e){
    document.querySelector('#order_msg').textContent = 'Error: ' + e.message;
  }
}

async function trackOrder(){
  const store = getToken(); if(!store){ alert('Falta Store Token'); return; }
  const oid = document.querySelector('#oid').value.trim(); if(!oid){ alert('Falta Order ID'); return; }
  const r = await fetch('/orders/' + encodeURIComponent(oid) + '?store=' + encodeURIComponent(store));
  const j = await r.json();
  if(!r.ok){ document.querySelector('#track_out').textContent = j.detail || 'Error'; return; }
  document.querySelector('#track_out').textContent = JSON.stringify(j, null, 2);
}

async function saveShipment(){
  const store = getToken(); if(!store){ alert('Falta Store Token'); return; }
  const order_id = document.querySelector('#ship_oid').value.trim();
  const carrier = document.querySelector('#carrier').value.trim();
  const tracking = document.querySelector('#tracking').value.trim();
  const eta = document.querySelector('#eta').value.trim();
  if(!order_id || !carrier || !tracking){
    alert('Completa Order ID, Transportadora y Tracking');
    return;
  }
  document.querySelector('#ship_msg').textContent = 'Guardando…';
  try{
    const r = await fetch('/shipment', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({store, order_id, carrier, tracking, eta})
    });
    const j = await r.json();
    if(!r.ok) throw new Error(j.detail || 'Error');
    document.querySelector('#ship_msg').textContent = '✔ Envío guardado';
  }catch(e){
    document.querySelector('#ship_msg').textContent = 'Error: ' + e.message;
  }
}
</script>
</body></html>
"""
    html = html.replace("__APP_NAME__", APP_NAME).replace("__SA_EMAIL__", sa_email)
    return HTMLResponse(html)

