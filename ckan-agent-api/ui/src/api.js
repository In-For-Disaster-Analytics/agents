// Thin API client for the CKAN agent. baseUrl defaults to "" so the Vite dev proxy handles /v1.

function authHeaders(jwt) {
  return jwt ? { Authorization: `Bearer ${jwt}` } : {};
}

export async function login(username, password, { baseUrl = "" } = {}) {
  const resp = await fetch(`${baseUrl}/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `Login failed (${resp.status})`);
  }
  return resp.json(); // { token, username, expires_in }
}

export async function uploadFiles(files, { baseUrl = "", jwt = "" } = {}) {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  const resp = await fetch(`${baseUrl}/v1/uploads`, {
    method: "POST",
    headers: { ...authHeaders(jwt) },
    body: form,
  });
  if (!resp.ok) throw new Error(`Upload failed (${resp.status}): ${await resp.text()}`);
  return resp.json();
}

export async function chat({ messages, metadata, model = "ckan-registration-agent", baseUrl = "", jwt = "" }) {
  const resp = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(jwt) },
    body: JSON.stringify({ model, messages, metadata, stream: false }),
  });
  if (!resp.ok) throw new Error(`Chat failed (${resp.status}): ${await resp.text()}`);
  const data = await resp.json();
  return data?.choices?.[0]?.message?.content ?? JSON.stringify(data);
}
