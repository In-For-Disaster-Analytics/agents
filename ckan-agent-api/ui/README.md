# CKAN Registration Agent — Chat UI

A light React + Vite chat window for the CKAN registration agent: attach files or a `.zip`,
chat, and let the agent review the files and propose CKAN metadata.

## Run (dev)

```bash
cd ui
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/v1/*` to the API at `http://localhost:8787` (override with
`VITE_API_TARGET`). Start the API first:

```bash
# from ckan-agent-api/
uvicorn app.main:app --reload --port 8787
```

Enable the agent in the API's `.env`: `CKAN_PERSONA_CHAT=1` and (for tool-based file review)
`CKAN_PERSONA_TOOLS=1`.

## Use

1. (Optional) Open **Settings** and paste your **CKAN JWT** — it's sent as
   `Authorization: Bearer <jwt>` on every request and used to talk to CKAN. It is stored only in
   your browser's localStorage, never sent anywhere else.
2. Click **📎** to attach files or a `.zip` (the zip is unpacked server-side).
3. Describe the dataset and **Send**. The agent picks a schema, reviews the files, asks any
   genuinely-missing fields one at a time, and proposes metadata. Reply inline to its questions.
4. Ask for a `dry run`, then `REGISTER` to apply (requires CKAN auth + dry-run first).

**New conversation** starts a fresh thread (new `session_id`).

## Build

```bash
npm run build        # outputs static assets to ui/dist/
```

`dist/` can be served by any static host (or wired into FastAPI later).
