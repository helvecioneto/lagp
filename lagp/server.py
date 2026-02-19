import os
import json
import logging
import secrets
import hashlib
import base64
import webbrowser
import asyncio
import time
from urllib.parse import urlencode, urlparse, parse_qs
from typing import Optional, Dict, Any
from datetime import datetime
import uvicorn
import httpx
import aiohttp
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel

# --- ANSI Colors ---
CYAN = "\033[96m"; GREEN = "\033[92m"; BOLD = "\033[1m"
RESET = "\033[0m"; YELLOW = "\033[93m"; RED = "\033[91m"; DIM = "\033[2m"

class _ColorFormatter(logging.Formatter):
    _LEVEL_COLORS = {
        logging.DEBUG:    DIM,
        logging.INFO:     CYAN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        ts    = self.formatTime(record, "%H:%M:%S")
        color = self._LEVEL_COLORS.get(record.levelno, CYAN)
        tag   = getattr(record, "tag", record.levelname[:5])
        return f"{DIM}{ts}{RESET}  {color}{BOLD}[{tag:<6}]{RESET}  {record.getMessage()}"

_logger = logging.getLogger("lagp")

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info":  logging.INFO,
    "ok":    logging.INFO,
    "warn":  logging.WARNING,
    "error": logging.ERROR,
}

def _log(tag: str, msg: str, level: str = "info") -> None:
    _logger.log(_LOG_LEVELS.get(level, logging.INFO), msg, extra={"tag": tag})

def _configure_logging(level_name: str = "INFO") -> None:
    level   = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter())
    _logger.setLevel(level)
    _logger.handlers  = [handler]
    _logger.propagate = False
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers  = [handler]
        lg.setLevel(level)
        lg.propagate = False

# --- Configuration ---
PORT = 11434
CALLBACK_PORT = 1455
CALLBACK_HOST = "localhost"
NO_BROWSER = False
SYNC_TO = ""
SYNC_SECRET = ""
API_KEYS: set = set()  # Empty = open access; non-empty = key required
CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
AUTHORIZE_URL = 'https://auth.openai.com/oauth/authorize'
TOKEN_URL = 'https://auth.openai.com/oauth/token'
REDIRECT_URI = f'http://localhost:{CALLBACK_PORT}/auth/callback'
OAUTH_SCOPE = 'openid profile email offline_access'
def get_auth_file() -> str:
    config_dir = Path.home() / ".lagp"
    config_dir.mkdir(parents=True, exist_ok=True)
    return str(config_dir / "auth.json")

# --- Global State ---
auth_state: Dict[str, Any] = {
    "verifier": None,
    "challenge": None,
    "state": None,
    "tokens": None
}

app = FastAPI(title="Local Use Chatgpt", description="A local proxy for OpenAI with OAuth", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def load_tokens():
    auth_file = get_auth_file()
    if os.path.exists(auth_file):
        try:
            with open(auth_file, 'r') as f:
                data = json.load(f)
                auth_state["tokens"] = data.get("tokens")
        except Exception as e:
            _log("TOKEN", f"error loading: {e}", level="error")

def save_tokens(tokens):
    auth_state["tokens"] = tokens
    try:
        with open(get_auth_file(), 'w') as f:
            json.dump({"tokens": tokens}, f, indent=2)
    except Exception as e:
        _log("TOKEN", f"error saving: {e}", level="error")

def generate_pkce():
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('utf-8')
    return verifier, challenge

def extract_account_id(token: str) -> Optional[str]:
    try:
        # Decode without verification just to get the payload
        # JWT header.payload.signature
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        # Add padding if needed
        payload_part = parts[1]
        padding = '=' * (4 - len(payload_part) % 4)
        payload_json = base64.urlsafe_b64decode(payload_part + padding).decode('utf-8')
        payload = json.loads(payload_json)
        
        # Try to find account ID in known locations
        if 'https://api.openai.com/auth' in payload:
            auth_claim = payload['https://api.openai.com/auth']
            return auth_claim.get('chatgpt_account_id') or auth_claim.get('account_id')
        
        return payload.get('account_id')
    except Exception as e:
        _log("TOKEN", f"error extracting account ID: {e}", level="error")
        return None

async def refresh_access_token():
    tokens = auth_state.get("tokens")
    if not tokens or not tokens.get("refresh_token"):
        _log("TOKEN", "no refresh token available", level="warn")
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TOKEN_URL,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens["refresh_token"],
                    "client_id": CLIENT_ID,
                }
            )
            
            if response.status_code == 200:
                new_tokens = response.json()
                # Merge with existing tokens to keep what's not returned (if any)
                updated_tokens = tokens.copy()
                updated_tokens.update(new_tokens)
                
                # If account_id is not in response, try to extract from new access_token
                if "access_token" in new_tokens:
                     aid = extract_account_id(new_tokens["access_token"])
                     if aid:
                         updated_tokens["account_id"] = aid

                save_tokens(updated_tokens)
                return updated_tokens["access_token"]
            else:
                _log("TOKEN", f"refresh failed: {response.text}", level="warn")
                return None
    except Exception as e:
        _log("TOKEN", f"refresh error: {e}", level="error")
        return None

async def get_valid_token():
    if not auth_state["tokens"]:
        load_tokens()
    
    tokens = auth_state.get("tokens")
    if not tokens or not tokens.get("access_token"):
        return None

    # Check expiry? Simple check: just return it, if it fails downstream with 401, we handle that
    # detailed expiry check would require parsing the JWT "exp" claim
    
    # Let's prevent unnecessary refreshes by checking exp if possible, 
    # but for simplicity, we'll implement a reactive refresh logic in the proxy.
    return tokens["access_token"]


# --- API Key Validation ---

def check_api_key(request: Request) -> None:
    """Validate the client API key when API_KEYS is configured.

    If API_KEYS is empty, all requests are allowed (open access).
    Otherwise, the request must carry 'Authorization: Bearer <key>' with a
    key that is in the configured set.
    """
    if not API_KEYS:
        return  # No keys configured → open access
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        key = auth_header[7:].strip()
        if key in API_KEYS:
            return
    _log("APIKEY", "rejected request — invalid or missing API key", level="warn")
    raise HTTPException(
        status_code=401,
        detail={
            "error": {
                "message": "Invalid or missing API key. Pass 'Authorization: Bearer <key>'.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
    )


# --- Auth Routes ---

@app.get("/login")
async def login():
    verifier, challenge = generate_pkce()
    state = secrets.token_hex(16)
    
    auth_state["verifier"] = verifier
    auth_state["challenge"] = challenge
    auth_state["state"] = state
    
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": OAUTH_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "vars" 
    }
    
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"
    
    print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║      OpenWebCode OAuth Login Flow        ║{RESET}")
    print(f"{CYAN}{BOLD}╚══════════════════════════════════════════╝{RESET}")
    if NO_BROWSER:
        print(f"\n  {GREEN}1.{RESET} Abra a URL abaixo no seu browser")
        print(f"  {GREEN}2.{RESET} Log in with your OpenAI account")
        print(f"  {GREEN}3.{RESET} Token saved automatically\n")
        print(f"  {YELLOW}Abra no seu browser:{RESET}")
        print(f"  {YELLOW}{url}{RESET}\n")
    else:
        print(f"\n  {GREEN}1.{RESET} Browser will open automatically")
        print(f"  {GREEN}2.{RESET} Log in with your OpenAI account")
        print(f"  {GREEN}3.{RESET} Token saved automatically\n")
        print(f"  {YELLOW}If the browser does not open, visit:{RESET}")
        print(f"  {YELLOW}{url}{RESET}\n")
        webbrowser.open(url)
    
    step1_text = "Open — open the URL below in your browser" if NO_BROWSER else "Open — browser opened automatically"
    return HTMLResponse(content=f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenWebCode — Login</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117; color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      display: flex; justify-content: center; align-items: center; min-height: 100vh;
    }}
    .card {{
      background: #161b22; border: 1px solid #30363d; border-radius: 12px;
      padding: 40px 48px; max-width: 480px; width: 100%; text-align: center;
    }}
    .logo {{ font-size: 2rem; margin-bottom: 8px; }}
    h1 {{ font-size: 1.5rem; color: #f0f6fc; margin-bottom: 8px; }}
    .subtitle {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 32px; }}
    .steps {{ text-align: left; margin-bottom: 32px; }}
    .step {{
      display: flex; align-items: center; gap: 14px;
      padding: 12px 0; border-bottom: 1px solid #21262d;
    }}
    .step:last-child {{ border-bottom: none; }}
    .step-num {{
      background: #238636; color: #fff; border-radius: 50%;
      width: 28px; height: 28px; display: flex; align-items: center;
      justify-content: center; font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
    }}
    .step-text {{ font-size: 0.95rem; color: #c9d1d9; }}
    .btn {{
      display: inline-block; background: #238636; color: #fff;
      border: none; border-radius: 6px; padding: 12px 24px;
      font-size: 1rem; font-weight: 600; text-decoration: none;
      cursor: pointer; transition: background 0.2s;
    }}
    .btn:hover {{ background: #2ea043; }}
    .waiting {{
      margin-top: 20px; color: #8b949e; font-size: 0.85rem;
    }}
    .dots::after {{
      content: ''; animation: dots 1.5s steps(4, end) infinite;
    }}
    @keyframes dots {{
      0%   {{ content: ''; }}
      25%  {{ content: '.'; }}
      50%  {{ content: '..'; }}
      75%  {{ content: '...'; }}
      100% {{ content: ''; }}
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🔐</div>
    <h1>OpenWebCode Login</h1>
    <p class="subtitle">OAuth authentication with your OpenAI account</p>
    <div class="steps">
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-text">{step1_text}</div>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <div class="step-text">Approve — log in with your OpenAI account</div>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <div class="step-text">Close — token saved, proxy is ready!</div>
      </div>
    </div>
    <a href="{url}" class="btn" target="_blank">Open Login Page</a>
    <p class="waiting">Waiting for authentication<span class="dots"></span></p>
  </div>
</body>
</html>""")

@app.get("/auth/callback") # This logic handles what the callback server would do
async def auth_callback(code: str, state: str):
    if state != auth_state["state"]:
        return HTMLResponse("State mismatch error", status_code=400)
    
    verifier = auth_state["verifier"]
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            }
        )
        
        if response.status_code != 200:
            return HTMLResponse(f"Token exchange failed: {response.text}", status_code=400)
            
        tokens = response.json()
        
        # Extract Account ID
        aid = extract_account_id(tokens.get("access_token"))
        if aid:
            tokens["account_id"] = aid
            
        save_tokens(tokens)
        
    return HTMLResponse("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Authentication Successful</title>
  <style>
    body { background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
           display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0; }
    .card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:48px; text-align:center; max-width:400px; }
    .check { font-size:4rem; margin-bottom:16px; }
    h1 { color:#2ea043; font-size:1.5rem; margin-bottom:8px; }
    p { color:#8b949e; font-size:0.95rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="check">✅</div>
    <h1>Authentication Successful</h1>
    <p>Token saved successfully.<br>You can close this tab.</p>
  </div>
</body>
</html>""")

@app.post("/auth/sync")
async def receive_sync_tokens(request: Request):
    if SYNC_SECRET:
        provided = request.headers.get("X-Sync-Secret", "")
        if provided != SYNC_SECRET:
            raise HTTPException(status_code=401, detail="Invalid sync secret")
    data = await request.json()
    tokens = data.get("tokens")
    if not tokens:
        raise HTTPException(status_code=400, detail="No tokens provided")
    save_tokens(tokens)
    _log("SYNC", "tokens received and saved", level="ok")
    return {"status": "ok"}

# Since the callback handles /auth/callback on port 1455 in the user code, 
# but FastAPI is running on 11434, we actually need a SECOND server or 
# just run THIS server on a port that can handle the callback if we want (but redirect URI is hardcoded).
# Wait, the user has REDIRECT_URI = http://localhost:1455/auth/callback
# So we need a listener on 1455.
# We can spin up a specific listener or jsut run the whole app on 1455? 
# The user wants "curl 127.0.0.1:11434". So we need two ports.
# We will start a separate background task or server for the callback?
# OR we can just tell the user to use the generated link, but the link points to localhost:1455.
# For simplicity, we can't easily spawn another process from here within FastAPI cleanly without managing it.
# However, we can use `uvicorn` to run on multiple ports? No.
#
# Solution: We will implement a small specialized callback server that runs ONLY during login, 
# similar to the Node.js implementation, OR we can just run the main server on 11434 
# and tell the user "We need to capture port 1455 for a moment". 
# Actually, the implementation plan said: "Callback Handler matching vars (port 1455)".
# Let's use a background thread to handle the callback on port 1455.

from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import platform
import argparse
from pathlib import Path

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/auth/callback':
            query = parse_qs(parsed.query)
            code = query.get('code', [None])[0]
            state = query.get('state', [None])[0]
            
            if code and state:
                # Send back to main app logic? 
                # Just do the exchange here for simplicity to avoid IPC complexity
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.exchange_code(code, state))
                loop.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Authentication Successful</title>
  <style>
    body { background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
           display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0; }
    .card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:48px; text-align:center; max-width:400px; }
    .check { font-size:4rem; margin-bottom:16px; }
    h1 { color:#2ea043; font-size:1.5rem; margin-bottom:8px; }
    p { color:#8b949e; font-size:0.95rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="check">&#x2705;</div>
    <h1>Authentication Successful</h1>
    <p>Token saved successfully.<br>You can close this tab.</p>
  </div>
</body>
</html>""")
                
                # Signal to verify/print
                print(f"\n{GREEN}{BOLD}╔══════════════════════════════════════════╗{RESET}")
                print(f"{GREEN}{BOLD}║   Login Successful! Token Acquired.      ║{RESET}")
                print(f"{GREEN}{BOLD}╚══════════════════════════════════════════╝{RESET}")
                print(f"\n  {GREEN}Proxy ready on port {PORT}{RESET}\n")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code or state")
        else:
            self.send_response(404)
            self.end_headers()

    async def exchange_code(self, code, state):
        if state != auth_state["state"]:
            print("State mismatch!")
            return

        verifier = auth_state["verifier"]
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                TOKEN_URL,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                data={
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": REDIRECT_URI,
                }
            )
            if resp.status_code == 200:
                tokens = resp.json()
                 # Extract Account ID
                aid = extract_account_id(tokens.get("access_token"))
                if aid:
                    tokens["account_id"] = aid
                save_tokens(tokens)
                if SYNC_TO:
                    await _sync_tokens(tokens)
            else:
                _log("AUTH", f"callback exchange failed: {resp.text}", level="error")

async def _sync_tokens(tokens: dict) -> bool:
    """Push tokens to a remote LAGP instance via /auth/sync. Returns True on success."""
    headers = {"Content-Type": "application/json"}
    if SYNC_SECRET:
        headers["X-Sync-Secret"] = SYNC_SECRET
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{SYNC_TO}/auth/sync", json={"tokens": tokens}, headers=headers)
            if r.status_code == 200:
                print(f"\n{GREEN}{BOLD}✓ Tokens synced to remote server.{RESET}")
                return True
            else:
                print(f"\n{YELLOW}✗ Sync failed: HTTP {r.status_code} — {r.text}{RESET}")
    except Exception as e:
        msg = str(e) or repr(e)
        print(f"\n{RED}✗ Sync error ({type(e).__name__}): {msg}{RESET}")
        _log("SYNC", f"sync to {SYNC_TO} failed — {type(e).__name__}: {msg}", level="error")
    return False


def start_callback_server(bind_host='127.0.0.1'):
    server = HTTPServer((bind_host, CALLBACK_PORT), OAuthCallbackHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()



# --- Proxy Logic ---

# --- CODEX/OAuth Proxy Logic ---

CODEX_URL = 'https://chatgpt.com/backend-api/codex/responses'
CODEX_MODELS_URL = 'https://chatgpt.com/backend-api/codex/models'
CODEX_CLIENT_VERSION = '1.0.0'

# Aliases: maps common OpenAI model names → real Codex slugs.
# Unknown names are passed through as-is (e.g. user specifies "gpt-5.3-codex" directly).
MODEL_MAP = {
    'gpt-4o-mini': 'gpt-5.1-codex-mini',
    'gpt-4o':             'gpt-5.3-codex',
    'gpt-4':              'gpt-5.3-codex',
    'gpt-4-turbo':        'gpt-5.3-codex',
    'gpt-4-turbo-preview':'gpt-5.3-codex',
    'gpt-40':             'gpt-5.3-codex',
    'gpt-5':              'gpt-5.3-codex',
}

# Populated at startup from /backend-api/codex/models
_available_models: list = []

def resolve_codex_model(model):
    return MODEL_MAP.get(model, model)

async def fetch_codex_models() -> list:
    """Query the ChatGPT backend for available Codex models."""
    token = await get_valid_token()
    if not token:
        return []
    tokens = auth_state.get("tokens", {})
    account_id = (tokens or {}).get("account_id") or extract_account_id(token)
    if not account_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                CODEX_MODELS_URL,
                params={"client_version": CODEX_CLIENT_VERSION},
                headers={
                    "Authorization": f"Bearer {token}",
                    "chatgpt-account-id": account_id,
                    "OpenAI-Beta": "responses=experimental",
                    "originator": "vars",
                    "User-Agent": "vars (macOS; arm64)",
                    "Accept": "application/json",
                },
            )
            if r.status_code == 200:
                return r.json().get("models", [])
            _log("MODEL", f"fetch failed HTTP {r.status_code}", level="warn")
    except Exception as e:
        _log("MODEL", f"fetch error: {e}", level="error")
    return []

def _map_content_types(role: str, content):
    """Convert OpenAI content types to Codex Responses API types.

    OpenAI Chat: type="text" | type="image_url"
    Codex API:   type="input_text" | type="input_image" | type="output_text"
    """
    if not isinstance(content, list):
        return content
    mapped = []
    for item in content:
        if not isinstance(item, dict):
            mapped.append(item)
            continue
        item = dict(item)
        t = item.get("type")
        if t == "text":
            item["type"] = "output_text" if role == "assistant" else "input_text"
        elif t == "image_url":
            item["type"] = "input_image"
            # aiohttp-style: Codex expects {"type":"input_image","image_url":...}
        mapped.append(item)
    return mapped


def convert_messages_to_codex_input(messages):
    """Convert OpenAI chat messages to Codex Responses API input format.

    Codex API only accepts role=user and role=assistant.
    - role=system  → handled separately as instructions
    - role=tool    → skipped (Codex doesn't support tool results)
    - role=function → skipped
    - assistant with content=null + tool_calls → converted to placeholder text
    """
    result = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        # Codex API only supports user/assistant roles
        if role in ("tool", "function"):
            continue
        content = m.get("content")
        if content is None:
            # Assistant message that only has tool_calls — convert to placeholder
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                names = ", ".join(
                    tc.get("function", {}).get("name", "tool")
                    for tc in tool_calls if isinstance(tc, dict)
                )
                content = f"[Called tool: {names}]"
            else:
                continue
        # Only pass role and content to avoid unknown fields confusing the Codex API
        result.append({"role": role, "content": _map_content_types(role, content)})
    return result

def extract_system_prompt(messages):
    for m in messages:
        if m.get("role") == "system":
            return m.get("content")
    return "You are a helpful assistant."

async def proxy_codex_request(token, body_json):
    """
    Handle request transformation for Codex Responses API
    """
    account_id = None
    tokens = auth_state.get("tokens", {})
    if tokens and tokens.get("account_id"):
        account_id = tokens["account_id"]
    
    if not account_id:
        # Try to extract again if missing
        account_id = extract_account_id(token)
        
    if not account_id:
         raise HTTPException(status_code=400, detail="Account ID not found in token")

    # Transform body
    original_model = body_json.get("model", "gpt-4o")
    codex_model = resolve_codex_model(original_model)
    messages = body_json.get("messages", [])
    client_expects_stream = body_json.get("stream", False)
    
    system_prompt = extract_system_prompt(messages)
    codex_input = convert_messages_to_codex_input(messages)
    
    # Build Codex Body
    codex_body = {
        "model": codex_model,
        "store": False,
        "stream": True, # Always stream from Codex side
        "instructions": system_prompt,
        "input": codex_input
    }
    
    # Headers
    headers = {
        'Authorization': f'Bearer {token}',
        'chatgpt-account-id': account_id,
        'OpenAI-Beta': 'responses=experimental',
        'originator': 'vars',
        'User-Agent': 'vars (macOS; arm64)', # Mimic vars
        'accept': 'text/event-stream',
        'content-type': 'application/json',
        'accept-encoding': 'identity',  # disable gzip to avoid httpx sync/async stream mismatch
    }
    
    model_info = f"{GREEN}{codex_model}{RESET}" if original_model == codex_model else f"{DIM}{original_model}{RESET} → {GREEN}{codex_model}{RESET}"
    stream_info = f"{GREEN}stream=on{RESET}" if client_expects_stream else f"{DIM}stream=off{RESET}"
    _log("OPENAI", f"chat  {model_info}  {stream_info}")
    
    async def stream_generator():
        _timeout = aiohttp.ClientTimeout(total=120)
        try:
            async with aiohttp.ClientSession(timeout=_timeout) as sess:
                async with sess.post(CODEX_URL, headers=headers, json=codex_body) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _log("OPENAI", f"HTTP {resp.status}: {error_text}", level="error")
                        return
                    buf = b""
                    async for raw_chunk in resp.content.iter_any():
                        buf += raw_chunk
                        while b"\n" in buf:
                            raw_line, buf = buf.split(b"\n", 1)
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                yield None
                                return
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "response.output_text.delta":
                                    content = data.get("delta", "")
                                    if content:
                                        yield {
                                            "id": "chatcmpl-codex",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": codex_model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": content},
                                                "finish_reason": None
                                            }]
                                        }
                                elif data.get("type") == "response.failed":
                                    _log("OPENAI", f"stream error: {data}", level="error")
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            _log("OPENAI", f"proxy error: {e}", level="error")
            raise

    if client_expects_stream:
        async def sse_wrapper():
            sent_done = False
            async for chunk in stream_generator():
                if chunk is None:
                    sent_done = True
                    stop_chunk = {
                        "id": "chatcmpl-codex",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": codex_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(stop_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    yield f"data: {json.dumps(chunk)}\n\n"
            if not sent_done:
                stop_chunk = {
                    "id": "chatcmpl-codex",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": codex_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(sse_wrapper(), media_type="text/event-stream")
    else:
        full_content = ""
        try:
            async for chunk in stream_generator():
                if chunk:
                    full_content += chunk["choices"][0]["delta"].get("content", "")
        except Exception as e:
            _log("OPENAI", f"proxy error: {e}", level="error")
            raise HTTPException(status_code=500, detail=str(e))
        return JSONResponse(content={
            "id": "chatcmpl-codex-full",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": codex_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })


@app.post("/api/chat")
async def handle_ollama_chat(request: Request):
    """
    Handle Ollama-style chat requests (POST /api/chat).
    Transforms them to Codex requests and adapts the response to Ollama's NDJSON format.
    """
    # 0. Client API key check
    check_api_key(request)

    # 1. Authentication
    token = await get_valid_token()
    if not token:
         raise HTTPException(status_code=401, detail="Authentication required")

    # 2. Parse Request
    try:
        body_json = await request.json()
    except Exception:
         raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 3. Prepare Codex Request
    account_id = None
    tokens = auth_state.get("tokens", {})
    if tokens and tokens.get("account_id"):
        account_id = tokens["account_id"]
    if not account_id:
        account_id = extract_account_id(token)
    if not account_id:
         raise HTTPException(status_code=400, detail="Account ID not found")

    messages = body_json.get("messages", [])
    original_model = body_json.get("model", "gpt-4o")
    # Force a compatible Codex model
    codex_model = resolve_codex_model(original_model)
    
    system_prompt = extract_system_prompt(messages)
    codex_input = convert_messages_to_codex_input(messages)
    
    codex_body = {
        "model": codex_model,
        "store": False,
        "stream": True,
        "instructions": system_prompt,
        "input": codex_input
    }
    
    headers = {
        'Authorization': f'Bearer {token}',
        'chatgpt-account-id': account_id,
        'OpenAI-Beta': 'responses=experimental',
        'originator': 'vars',
        'User-Agent': 'vars (macOS; arm64)',
        'accept': 'text/event-stream',
        'content-type': 'application/json',
        'accept-encoding': 'identity',  # disable gzip to avoid httpx sync/async stream mismatch
    }

    model_info = f"{GREEN}{codex_model}{RESET}" if original_model == codex_model else f"{DIM}{original_model}{RESET} → {GREEN}{codex_model}{RESET}"
    _log("OLLAMA", f"chat  {model_info}  {GREEN}stream=on{RESET}")
    
    async def ollama_generator():
        done_sent = False
        buf = b""
        _timeout = aiohttp.ClientTimeout(total=120)
        try:
            async with aiohttp.ClientSession(timeout=_timeout) as sess:
                async with sess.post(CODEX_URL, headers=headers, json=codex_body) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _log("OLLAMA", f"HTTP {resp.status}: {error_text}", level="error")
                        return
                    async for raw_chunk in resp.content.iter_any():
                        buf += raw_chunk
                        while b"\n" in buf:
                            raw_line, buf = buf.split(b"\n", 1)
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                done_sent = True
                                final_msg = {
                                    "model": original_model,
                                    "created_at": datetime.now().isoformat() + "Z",
                                    "message": {"role": "assistant", "content": ""},
                                    "done_reason": "stop",
                                    "done": True,
                                    "total_duration": 0,
                                    "load_duration": 0,
                                    "prompt_eval_count": 0,
                                    "eval_count": 0
                                }
                                yield json.dumps(final_msg) + "\n"
                                return
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "response.output_text.delta":
                                    content = data.get("delta", "")
                                    if content:
                                        msg = {
                                            "model": original_model,
                                            "created_at": datetime.now().isoformat() + "Z",
                                            "message": {"role": "assistant", "content": content},
                                            "done": False
                                        }
                                        yield json.dumps(msg) + "\n"
                            except (json.JSONDecodeError, KeyError):
                                continue
                    if not done_sent:
                        final_msg = {
                            "model": original_model,
                            "created_at": datetime.now().isoformat() + "Z",
                            "message": {"role": "assistant", "content": ""},
                            "done_reason": "stop",
                            "done": True,
                            "total_duration": 0,
                            "load_duration": 0,
                            "prompt_eval_count": 0,
                            "eval_count": 0
                        }
                        yield json.dumps(final_msg) + "\n"
        except Exception as e:
            _log("OLLAMA", f"proxy exception: {e}", level="error")

    return StreamingResponse(ollama_generator(), media_type="application/x-ndjson")


@app.get("/api/tags")
async def list_ollama_models(request: Request):
    """Ollama-compatible model discovery."""
    check_api_key(request)
    if _available_models:
        slugs = [m["slug"] for m in _available_models if m.get("slug")]
    else:
        slugs = list(MODEL_MAP.keys())
    models = [
        {
            "name": slug,
            "model": slug,
            "modified_at": "2024-05-13T00:00:00Z",
            "size": 0,
            "digest": "sha256:" + "0" * 64,
            "details": {"format": "gguf", "family": "gpt", "families": ["gpt"],
                        "parameter_size": "0B", "quantization_level": "Q4_0"},
        }
        for slug in slugs
    ]
    return {"models": models}

@app.get("/v1/models")
async def list_openai_models(request: Request):
    """Return available models in OpenAI format (used by Cline, Cursor, etc.)"""
    check_api_key(request)
    if _available_models:
        data = [
            {
                "id": m["slug"],
                "object": "model",
                "created": 1715000000,
                "owned_by": "openai",
                "permission": [],
                "root": m["slug"],
                "parent": None,
            }
            for m in _available_models if m.get("slug")
        ]
    else:
        data = [
            {"id": mid, "object": "model", "created": 1715000000, "owned_by": "openai",
             "permission": [], "root": mid, "parent": None}
            for mid in MODEL_MAP
        ]
    return JSONResponse({"object": "list", "data": data})


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    """OpenAI-compatible chat completions — proxied to the Codex API."""
    check_api_key(request)
    token = await get_valid_token()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        body_json = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    return await proxy_codex_request(token, body_json)


@app.post("/v1/responses")
async def handle_responses_api(request: Request):
    """OpenAI Responses API — passed through directly to the Codex backend.

    The Codex API already speaks the Responses API SSE format, so we just
    swap the URL and inject the auth headers, then stream bytes straight back.
    """
    check_api_key(request)
    token = await get_valid_token()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    tokens = auth_state.get("tokens", {})
    account_id = (tokens or {}).get("account_id") or extract_account_id(token)
    if not account_id:
        raise HTTPException(status_code=400, detail="Account ID not found in token")

    try:
        body_json = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    original_model = body_json.get("model", "gpt-4o")
    codex_model = resolve_codex_model(original_model)
    body = dict(body_json)
    body["model"] = codex_model

    headers = {
        'Authorization': f'Bearer {token}',
        'chatgpt-account-id': account_id,
        'OpenAI-Beta': 'responses=experimental',
        'originator': 'vars',
        'User-Agent': 'vars (macOS; arm64)',
        'accept': 'text/event-stream',
        'content-type': 'application/json',
        'accept-encoding': 'identity',
    }

    model_info = f"{GREEN}{codex_model}{RESET}" if original_model == codex_model else f"{DIM}{original_model}{RESET} → {GREEN}{codex_model}{RESET}"
    _log("OPENAI", f"responses  {model_info}")

    async def _passthrough():
        _timeout = aiohttp.ClientTimeout(total=120)
        try:
            async with aiohttp.ClientSession(timeout=_timeout) as sess:
                async with sess.post(CODEX_URL, headers=headers, json=body) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _log("OPENAI", f"responses HTTP {resp.status}: {error_text}", level="error")
                        return
                    async for chunk in resp.content.iter_any():
                        if chunk:
                            yield chunk
        except Exception as e:
            _log("OPENAI", f"responses proxy error: {e}", level="error")

    return StreamingResponse(_passthrough(), media_type="text/event-stream")


_OPEN_PATHS = {"login", "auth/callback", "auth/sync", "docs", "openapi.json", "favicon.ico", ""}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_all(request: Request, path: str):
    # Dedicated handlers above cover these paths; this catch-all won't see them,
    # but keep the guard as a safety net.
    if path in {"login", "auth/callback", "docs", "openapi.json",
                "api/tags", "api/chat", "v1/models",
                "v1/chat/completions", "v1/responses"}:
        raise HTTPException(status_code=404, detail="Not found")

    if path == "favicon.ico":
        return Response(status_code=404)

    if path == "":
        return HTMLResponse("<h1>Local LLM Proxy</h1><p>Status: Running. <a href='/login'>Login</a></p>")

    if request.method == "OPTIONS":
        return Response(status_code=204)

    # Client API key check (skip for internal auth/admin paths)
    if path not in _OPEN_PATHS:
        check_api_key(request)

    # Load token
    token = await get_valid_token()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required. Please visit /login to authenticate.")

    # Check if this is a chat completion request that we should intercept for Codex
    # Path might be v1/chat/completions
    if "chat/completions" in path and request.method == "POST":
         # We assume usage of Codex tokens means we use Codex API
         # Verify if token is JWT (contains dots)
         if "." in token:
             try:
                 body = await request.json()
                 return await proxy_codex_request(token, body)
             except Exception as e:
                 _log("OPENAI", f"error parsing request body: {e}", level="error")
                 # Fallthrough to standard proxy if parsing fails

    # --- Standard OpenAI Proxy (Forwarding) ---
    url = f"https://api.openai.com/{path}"
    
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length', 'connection', 'accept-encoding']}
    headers["Authorization"] = f"Bearer {token}"
    
    tokens = auth_state.get("tokens", {})
    if tokens and tokens.get("account_id"):
        headers["ChatGPT-Account-Id"] = tokens["account_id"]
    
    client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    
    try:
        req = client.build_request(
            request.method, 
            url, 
            headers=headers, 
            content=await request.body()
        )
        
        r = await client.send(req, stream=True)
        
        if r.status_code == 401:
            _log("TOKEN", "got 401, attempting token refresh...", level="warn")
            await r.aclose()
            new_token = await refresh_access_token()
            
            if new_token:
                headers["Authorization"] = f"Bearer {new_token}"
                req = client.build_request(request.method, url, headers=headers, content=await request.body())
                r = await client.send(req, stream=True)
            else:
                await client.aclose()
                return JSONResponse(status_code=401, content={
                    "error": {
                        "message": "Token expired. Please visit /login to re-authenticate.",
                        "type": "authentication_error",
                        "code": "token_expired"
                    }
                })

        return StreamingResponse(
            r.aiter_bytes(),
            status_code=r.status_code,
            headers=dict(r.headers),
            background=BackgroundTask(client.aclose)
        )
        
    except Exception as e:
        await client.aclose()
        _log("PROXY", f"error: {e}", level="error")
        raise HTTPException(status_code=500, detail=str(e))



def print_startup_banner():
    tokens = auth_state.get("tokens")
    authenticated = bool(tokens and tokens.get("access_token"))
    account_id = (tokens.get("account_id") or "")[:24] if tokens else ""
    server_url = f"http://{CALLBACK_HOST}:{PORT}" if CALLBACK_HOST != 'localhost' else f"http://localhost:{PORT}"

    if authenticated:
        status_line = f"  {GREEN}✓ Authenticated{RESET}"
    else:
        status_line = f"  {YELLOW}✗ Not authenticated{RESET}"

    if API_KEYS:
        apikey_line = f"  {GREEN}✓ {len(API_KEYS)} key(s) configured{RESET}"
    else:
        apikey_line = f"  {DIM}None (open access){RESET}"

    print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║                 LAGP   v1.0                  ║{RESET}")
    print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════╝{RESET}\n")

    print(f"  Status    :{status_line}")
    print(f"  API Keys  : {apikey_line}")
    if authenticated and account_id:
        print(f"  Account   : {account_id}")
    if _available_models:
        visible = [m["slug"] for m in _available_models if m.get("visibility") == "list"]
        print(f"\n  Models:")
        for slug in visible:
            print(f"    {GREEN}●{RESET} {slug}")
        print(f"\n  {CYAN}→ Live list:{RESET} curl {server_url}/v1/models")

    print(f"\n  Listening : {server_url}\n")
    print(f"  Endpoints:")
    print(f"    GET  /login                OAuth login flow")
    print(f"    GET  /v1/models            List available models (JSON)")
    print(f"    POST /v1/chat/completions  OpenAI-compatible chat  (Cline, Cursor, etc.)")
    print(f"    POST /api/chat             Ollama-compatible chat")
    _key_hint = f'      -H "Authorization: Bearer <your-api-key>" \\\n' if API_KEYS else ""
    print(f"\n  Quick test:")
    print(f'    curl {server_url}/v1/chat/completions \\')
    print(f'      -H "Content-Type: application/json" \\')
    if _key_hint:
        print(f'      -H "Authorization: Bearer <your-api-key>" \\')
    print(f'      -d \'{{"model":"gpt-5.3-codex","messages":[{{"role":"user","content":"hi"}}]}}\'')
    print(f"\n{'─' * 48}")

    if not authenticated:
        # Remote server waiting for token sync
        if SYNC_SECRET and not SYNC_TO:
            print(f"  {YELLOW}→ Waiting for tokens via /auth/sync{RESET}")
            print(f"    On your local machine, run:")
            print(f"    {CYAN}lagp --sync-to http://<server-ip>:{PORT} --sync-secret <secret>{RESET}")
        # Local machine pushing tokens to remote after auth
        elif SYNC_TO:
            print(f"  {YELLOW}→ After login, tokens will be pushed to the remote server.{RESET}")
            print(f"    Or manually visit: {server_url}/login")
        # Browser opens automatically (local mode)
        elif not NO_BROWSER:
            print(f"  {YELLOW}→ Opening browser for login in 2 seconds...{RESET}")
            print(f"    Or manually visit: {server_url}/login")
        # No-browser fallback
        else:
            print(f"  {YELLOW}→ Open the URL below in your browser to authenticate:{RESET}")
            print(f"    {server_url}/login")

    print(f"\n{'─' * 48}")
    print(f"  {DIM}Running on a remote server?  See README for setup options.{RESET}")
    print()


def _open_login_browser():
    """Wait for the server to be ready, then open the login page."""
    time.sleep(2)
    server_url = f"http://{CALLBACK_HOST}:{PORT}" if CALLBACK_HOST != 'localhost' else f"http://localhost:{PORT}"
    webbrowser.open(f"{server_url}/login")


def main():
    global _available_models, PORT, CALLBACK_PORT, REDIRECT_URI, CALLBACK_HOST, NO_BROWSER, SYNC_TO, SYNC_SECRET, API_KEYS

    parser = argparse.ArgumentParser(description="LAGP — LLM Auth Gateway Proxy")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port for the proxy server (default: {PORT})")
    parser.add_argument("--callback-port", type=int, default=CALLBACK_PORT, help=f"Port for the OAuth callback server (default: {CALLBACK_PORT})")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], metavar="LEVEL", help="Log verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)")
    parser.add_argument("--no-browser", action="store_true", help="Não abrir browser automaticamente (útil para servidores headless/remotos)")
    parser.add_argument("--callback-host", default="localhost", metavar="HOST", help="Hostname/IP do servidor para o redirect OAuth (padrão: localhost). Defina como o hostname público do servidor ao rodar remotamente.")
    parser.add_argument("--sync-to", default="", metavar="URL", help="URL do servidor remoto para enviar tokens após autenticação (ex: http://meuservidor.com:11434)")
    parser.add_argument("--sync-secret", default="", metavar="SECRET", help="Secret compartilhado para proteger o endpoint /auth/sync no servidor remoto")
    parser.add_argument(
        "--api-key",
        action="append",
        dest="api_keys",
        metavar="KEY",
        default=[],
        help=(
            "API key aceita para autenticação dos clientes (pode ser especificado múltiplas vezes). "
            "Quando definida, requisições devem incluir 'Authorization: Bearer <key>'. "
            "Se não definida, qualquer chave (ou nenhuma) é aceita."
        ),
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)

    PORT = args.port
    CALLBACK_PORT = args.callback_port
    CALLBACK_HOST = args.callback_host
    NO_BROWSER = args.no_browser
    SYNC_TO = args.sync_to.rstrip('/')
    SYNC_SECRET = args.sync_secret
    API_KEYS = set(args.api_keys)
    REDIRECT_URI = f'http://{CALLBACK_HOST}:{CALLBACK_PORT}/auth/callback'

    if CALLBACK_HOST not in ('localhost', '127.0.0.1'):
        print(f"\n{YELLOW}{BOLD}⚠  Atenção:{RESET}{YELLOW} --callback-host '{CALLBACK_HOST}' pode falhar no OAuth.")
        print(f"   O provider OAuth só aceita 'localhost' como redirect_uri.")
        print(f"   Use --sync-to + --sync-secret para autenticação remota:{RESET}")
        print(f"   {DIM}Servidor: lagp --no-browser --sync-secret KEY{RESET}")
        print(f"   {DIM}Local:    lagp --sync-to http://{CALLBACK_HOST}:{PORT} --sync-secret KEY{RESET}\n")

    callback_bind = '0.0.0.0' if CALLBACK_HOST != 'localhost' else '127.0.0.1'
    start_callback_server(callback_bind)
    load_tokens()
    if auth_state.get("tokens"):
        if SYNC_TO:
            asyncio.run(_sync_tokens(auth_state["tokens"]))
        models = asyncio.run(fetch_codex_models())
        if models:
            _available_models = models
    print_startup_banner()
    if not auth_state.get("tokens") and not NO_BROWSER:
        threading.Thread(target=_open_login_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
