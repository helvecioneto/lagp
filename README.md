# LAGP — LLM Auth Gateway Proxy

> A local OpenAI-compatible proxy powered by your ChatGPT Plus subscription — no API key required.

[![PyPI version](https://img.shields.io/pypi/v/lagp.svg)](https://pypi.org/project/lagp/)
[![Python](https://img.shields.io/pypi/pyversions/lagp.svg)](https://pypi.org/project/lagp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is LAGP?

**LAGP** runs a local HTTP server on port `11434` that speaks the OpenAI API. Any tool that accepts a custom base URL — Cline, Cursor, Continue, Open WebUI, or your own scripts — can point to `http://localhost:11434` and use your existing ChatGPT Plus account as the backend, at no extra cost.

Authentication happens once through a standard browser OAuth flow. The token is stored locally at `~/.lagp/auth.json` and refreshed automatically.

```
Your tool (Cline, Cursor…)
        │  OpenAI API calls
        ▼
 http://localhost:11434   ← LAGP
        │  ChatGPT OAuth
        ▼
  chatgpt.com backend
```

---

## Requirements

- Python 3.9+
- An active **ChatGPT Plus** (or higher) subscription
- A modern browser (for the one-time OAuth login)

---

## Installation

### From PyPI

```bash
pip install lagp
```

### From source

```bash
git clone https://github.com/helvecioneto/lagp.git
cd lagp
pip install .
```

---

## Usage

Start the server:

```bash
lagp
```

On the first run, a browser window opens automatically for login. Sign in with your OpenAI account and the token is saved. The server starts immediately after authentication.

On subsequent runs, the saved token is loaded and the server starts right away.

### Configure your tool

Point any OpenAI-compatible client to the local server:

| Setting | Value |
|---|---|
| Base URL | `http://localhost:11434/v1` |
| API Key | any non-empty string (e.g. `lagp`) — or your configured key if using `--api-key` |

---

## API key protection

By default LAGP accepts any value (or no value) in the `Authorization` header — convenient for local, single-user setups.

When you expose LAGP on a network (LAN, VPS, etc.) you can restrict access to a fixed set of keys:

```bash
# Single key
lagp --api-key my-secret-key

# Multiple keys (repeat the flag)
lagp --api-key key-for-alice --api-key key-for-bob
```

Once at least one `--api-key` is passed, every request to the proxy **must** include a matching key:

```
Authorization: Bearer <your-api-key>
```

Requests with a missing or wrong key are rejected with HTTP **401**.

### Endpoints affected

| Endpoint | Protected |
|---|:---:|
| `POST /v1/chat/completions` | ✓ |
| `POST /api/chat` | ✓ |
| `GET /v1/models` | ✓ |
| `GET /api/tags` | ✓ |
| `GET /login`, `/auth/*` | — (always open) |

> **Tip:** generate a strong key with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

---

## Available models

LAGP queries the ChatGPT backend at startup and lists the models your account can access. To see the live list at any time:

```bash
curl http://localhost:11434/v1/models
```

Typical models available with ChatGPT Plus:

| Model | Notes |
|---|---|
| `gpt-5.3-codex` | Latest, recommended |
| `gpt-5.2-codex` | Previous generation |
| `gpt-5.1-codex-max` | High-capacity variant |
| `gpt-5.1-codex-mini` | Faster, lighter |
| `gpt-5.2` | General purpose |

> The list is fetched live from the API — new models appear automatically as OpenAI releases them.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/login` | Start OAuth login flow |
| `GET` | `/v1/models` | List available models (OpenAI format) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/api/chat` | Ollama-compatible chat (NDJSON) |
| `GET` | `/api/tags` | Ollama model discovery |

---

## Quick test

Check available models:

```bash
curl http://localhost:11434/v1/models
```

Send a chat message (open access — no `--api-key` configured):

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5.3-codex", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Send a chat message (with `--api-key` configured):

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"model": "gpt-5.3-codex", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Use with the OpenAI Python SDK:

```python
from openai import OpenAI

# Open access (no --api-key configured)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="lagp")

# With --api-key configured — pass the real key
client = OpenAI(base_url="http://localhost:11434/v1", api_key="my-secret-key")

response = client.chat.completions.create(
    model="gpt-5.3-codex",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Running on a remote server

LAGP works great on headless remote servers (VPS, cloud instances, etc.). Because the OAuth provider only allows `localhost` as a valid redirect URI, you cannot authenticate directly on a remote machine through its public IP. There are three supported approaches.

---

### Method 1 — Token sync (recommended)

Authenticate on your local machine and push the tokens automatically to the remote server over HTTP.

**On the remote server**, start LAGP in no-browser mode with a shared secret:

```bash
lagp --no-browser --sync-secret <your-secret>
```

**On your local machine**, run:

```bash
lagp --sync-to http://<server-ip>:11434 --sync-secret <your-secret>
```

A browser opens, you log in with your OpenAI account, and the tokens are pushed to the remote server automatically. From that point the remote server is fully authenticated.

**Network requirement:** port `11434` must be open on the remote server.

> **Note:** `--sync-secret` is strongly recommended. Without it, anyone who can reach port `11434` could push arbitrary tokens to your server.

---

### Method 2 — SSH tunnel

No configuration needed beyond SSH access. The tunnel makes the remote server's ports appear as local ports, so `localhost` redirect URIs work transparently.

**Open two tunnels from your local machine:**

```bash
ssh -L 11434:localhost:11434 -L 1455:localhost:1455 user@<server>
```

**On the remote server** (in another terminal or tmux):

```bash
lagp --no-browser
```

**On your local machine**, open in a browser:

```
http://localhost:11434/login
```

OAuth redirects to `localhost:1455`, which the tunnel forwards to the remote server. Tokens are saved on the server. Close the tunnel when done — `lagp` runs independently from that point.

**Network requirement:** only SSH (port 22) needs to be open.

---

### Method 3 — Copy auth.json manually

If you have already authenticated locally, just copy the token file to the remote server:

```bash
scp ~/.lagp/auth.json user@<server>:~/.lagp/auth.json
```

Then start LAGP on the remote server:

```bash
lagp --no-browser
```

It will load the existing token and start immediately. Tokens refresh automatically, so this file stays valid until you explicitly log out.

---

### Comparison

| | Method 1 (sync) | Method 2 (SSH tunnel) | Method 3 (copy) |
|---|:---:|:---:|:---:|
| Requires open port | 11434 | 22 only | 22 only |
| Fully automated | ✓ | — | — |
| No extra tooling | ✓ | needs SSH tunnel | needs SCP |
| Works without local LAGP | — | — | ✓ |

---

### Re-authentication

Tokens expire eventually. To re-authenticate on a remote server, run either Method 1 or Method 2 again. The new token overwrites the old one at `~/.lagp/auth.json`.

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--port PORT` | `11434` | Port for the proxy server |
| `--callback-port PORT` | `1455` | Port for the OAuth callback listener |
| `--no-browser` | off | Do not open a browser automatically |
| `--api-key KEY` | — | Accepted client API key (repeat for multiple keys). When set, all requests must include `Authorization: Bearer <key>` |
| `--sync-to URL` | — | Push tokens to a remote LAGP after login |
| `--sync-secret SECRET` | — | Shared secret protecting the `/auth/sync` endpoint |
| `--log-level LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Token storage

The auth token is stored at `~/.lagp/auth.json` on all platforms. To log out, delete this file and restart `lagp`.

---

## License

MIT — see [LICENSE](LICENSE).
