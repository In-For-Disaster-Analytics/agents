import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { chat, login, uploadFiles } from "./api.js";

const newSessionId = () =>
  (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2)).replace(/-/g, "");

const SESSIONS_KEY = "ckan.sessions";
const MAX_SESSIONS = 50;

function loadSessions() {
  try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]"); }
  catch { return []; }
}

function persistSession({ id, messages, uploadDir }) {
  if (!messages.length) return;
  const firstUser = messages.find((m) => m.role === "user");
  const title = firstUser ? firstUser.content.slice(0, 72) : new Date().toLocaleString();
  const all = loadSessions().filter((s) => s.id !== id);
  all.unshift({ id, title, savedAt: new Date().toISOString(), messages, uploadDir });
  if (all.length > MAX_SESSIONS) all.splice(MAX_SESSIONS);
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(all));
}

export default function App() {
  const [baseUrl, setBaseUrl] = useState(localStorage.getItem("ckan.baseUrl") || "");
  const [jwt, setJwt] = useState(localStorage.getItem("ckan.jwt") || "");
  const [username, setUsername] = useState(localStorage.getItem("ckan.username") || "");
  const [showSettings, setShowSettings] = useState(false);

  // Login form state
  const [loginUser, setLoginUser] = useState("");
  const [loginPass, setLoginPass] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [loginError, setLoginError] = useState("");

  const [sessionId, setSessionId] = useState(newSessionId());
  const [uploadDir, setUploadDir] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sessions, setSessions] = useState(loadSessions);
  const [showHistory, setShowHistory] = useState(false);

  const fileRef = useRef(null);
  const endRef = useRef(null);
  const passRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);
  useEffect(() => localStorage.setItem("ckan.jwt", jwt), [jwt]);
  useEffect(() => localStorage.setItem("ckan.baseUrl", baseUrl), [baseUrl]);
  useEffect(() => localStorage.setItem("ckan.username", username), [username]);
  useEffect(() => {
    if (messages.length) {
      persistSession({ id: sessionId, messages, uploadDir });
      setSessions(loadSessions());
    }
  }, [messages]);

  const add = (role, content) => setMessages((m) => [...m, { role, content }]);

  function resetConversation() {
    if (messages.length) persistSession({ id: sessionId, messages, uploadDir });
    setSessionId(newSessionId());
    setUploadDir("");
    setMessages([]);
    setError("");
  }

  function switchToSession(s) {
    if (messages.length) persistSession({ id: sessionId, messages, uploadDir });
    setSessionId(s.id);
    setMessages(s.messages || []);
    setUploadDir(s.uploadDir || "");
    setError("");
    setShowHistory(false);
  }

  function signOut() {
    setJwt("");
    setUsername("");
    localStorage.removeItem("ckan.jwt");
    localStorage.removeItem("ckan.username");
    resetConversation();
  }

  async function handleLogin(e) {
    e.preventDefault();
    setLoginError("");
    setLoginBusy(true);
    try {
      const result = await login(loginUser.trim(), loginPass, { baseUrl });
      setJwt(result.token);
      setUsername(result.username);
      setLoginPass("");
    } catch (err) {
      setLoginError(String(err.message || err));
    } finally {
      setLoginBusy(false);
    }
  }

  async function onFiles(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    setBusy(true);
    setError("");
    if (fileRef.current) fileRef.current.value = "";
    let newDir = "";
    try {
      const res = await uploadFiles(files, { baseUrl, jwt });
      newDir = res.dir || "";
      setUploadDir(newDir);
      const names = (res.files || []).map((f) => f.name);
      add("note", `Uploaded ${names.length} file(s): ${names.join(", ")}` + (res.warnings?.length ? `\n⚠ ${res.warnings.join("; ")}` : ""));
    } catch (e) {
      setError(String(e.message || e));
      setBusy(false);
      return;
    }
    // Still busy — auto-trigger metadata extraction so the user doesn't have to say anything.
    try {
      const reply = await chat({ messages: [], metadata: { session_id: sessionId, upload_dir: newDir }, baseUrl, jwt });
      add("assistant", reply);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setError("");
    add("user", text);
    const history = [...messages, { role: "user", content: text }]
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));
    const metadata = { session_id: sessionId };
    if (uploadDir) metadata.upload_dir = uploadDir;
    setBusy(true);
    try {
      const reply = await chat({ messages: history, metadata, baseUrl, jwt });
      add("assistant", reply);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  // ── Login screen ──────────────────────────────────────────────────────────
  if (!jwt) {
    return (
      <div className="app login-screen">
        <header>
          <h1>CKAN Registration Agent</h1>
          <div className="header-actions">
            <button onClick={() => setShowSettings((s) => !s)}>{showSettings ? "Hide" : "Settings"}</button>
          </div>
        </header>

        {showSettings && (
          <div className="settings">
            <label>
              API base URL (blank = dev proxy /v1)
              <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="e.g. http://localhost:8787" />
            </label>
          </div>
        )}

        <main className="login-container">
          <form className="login-form" onSubmit={handleLogin}>
            <h2>Sign in with Tapis</h2>
            <label>
              Username
              <input
                type="text"
                value={loginUser}
                onChange={(e) => setLoginUser(e.target.value)}
                autoComplete="username"
                autoFocus
                required
              />
            </label>
            <label>
              Password
              <input
                ref={passRef}
                type="password"
                value={loginPass}
                onChange={(e) => setLoginPass(e.target.value)}
                autoComplete="current-password"
                required
              />
            </label>
            {loginError && <div className="login-error">{loginError}</div>}
            <button type="submit" disabled={loginBusy || !loginUser.trim() || !loginPass}>
              {loginBusy ? "Signing in…" : "Sign in"}
            </button>
          </form>
        </main>
      </div>
    );
  }

  // ── Chat screen ───────────────────────────────────────────────────────────
  return (
    <div className="app">
      <header>
        <h1>CKAN Registration Agent</h1>
        <div className="header-actions">
          <span className="thread">thread: {sessionId.slice(0, 8)}</span>
          <span className="signed-in">{username}</span>
          <button onClick={signOut}>Sign out</button>
          <button onClick={resetConversation}>New</button>
          <button onClick={() => { setShowHistory((h) => !h); setShowSettings(false); }}>
            {showHistory ? "Hide history" : `History${sessions.length ? ` (${sessions.length})` : ""}`}
          </button>
          <button onClick={() => { setShowSettings((s) => !s); setShowHistory(false); }}>{showSettings ? "Hide" : "Settings"}</button>
        </div>
      </header>

      {showSettings && (
        <div className="settings">
          <label>
            API base URL (blank = dev proxy /v1)
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="e.g. http://localhost:8787" />
          </label>
        </div>
      )}

      {showHistory && (
        <div className="history-panel">
          {sessions.length === 0 ? (
            <div className="history-empty">No saved conversations yet.</div>
          ) : (
            <ul className="history-list">
              {sessions.map((s) => (
                <li
                  key={s.id}
                  className={`history-item${s.id === sessionId ? " active" : ""}`}
                  onClick={() => switchToSession(s)}
                >
                  <span className="history-title">{s.title}</span>
                  <span className="history-meta">
                    {s.id.slice(0, 8)} · {new Date(s.savedAt).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <main className="transcript">
        {messages.length === 0 && (
          <div className="empty">Attach files or a .zip — the agent will immediately start extracting metadata and propose CKAN fields for review.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="role">{m.role}</div>
            {m.role === "assistant"
              ? <div className="content md"><ReactMarkdown>{m.content}</ReactMarkdown></div>
              : <pre className="content">{m.content}</pre>}
          </div>
        ))}
        {busy && <div className="msg assistant"><div className="role">assistant</div><div className="content">…</div></div>}
        <div ref={endRef} />
      </main>

      {error && <div className="error">{error}</div>}
      {uploadDir && <div className="uploaddir">attached upload: <code>{uploadDir}</code></div>}

      <footer className="composer">
        <button className="attach" onClick={() => fileRef.current?.click()} disabled={busy} title="Attach files or a .zip">📎</button>
        <input ref={fileRef} type="file" multiple onChange={onFiles} style={{ display: "none" }} />
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Describe the dataset, or answer the agent's question…"
          rows={2}
        />
        <button className="send" onClick={send} disabled={busy || !input.trim()}>Send</button>
      </footer>
    </div>
  );
}
