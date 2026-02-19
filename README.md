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
| API Key | any non-empty string (e.g. `lagp`) |

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

Send a chat message:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5.3-codex", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Use with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="lagp")

response = client.chat.completions.create(
    model="gpt-5.3-codex",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Token storage

The auth token is stored at `~/.lagp/auth.json` on all platforms. To log out, delete this file and restart `lagp`.

---

## License

MIT — see [LICENSE](LICENSE).
