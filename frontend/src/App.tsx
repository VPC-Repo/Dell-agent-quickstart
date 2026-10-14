import { useEffect, useState } from "react";
import LiveDemo from "./LiveDemo";

export async function api(path: string, method = "GET", body?: any) {
  const token = sessionStorage.getItem("builder-token");
  const r = await fetch("/api" + path, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: "Bearer " + token } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({ detail: r.statusText }));
    throw Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  }
  return r;
}
export const get = async (path: string) => (await api(path)).json();

export default function App() {
  const [auth, setAuth] = useState(false);
  const [authRequired, setAuthRequired] = useState(false);
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  // Confirms the token (when sign-in is on) with an authorized call, then shows the demo.
  async function refresh() {
    await get("/catalog");
    setReady(true);
  }
  useEffect(() => {
    get("/health")
      .then(async (h) => {
        setAuthRequired(!!h.auth_required);
        if (h.auth_required && !sessionStorage.getItem("builder-token")) {
          setAuth(true);
          return;
        }
        await refresh();
      })
      .catch((e) => {
        setError(e.message);
        if (e.message.includes("token")) setAuth(true);
      });
  }, []);

  return (
    <div className="app mode-workspace">
      <div className="main">
        <LiveDemo
          api={api}
          get={get}
          ready={ready}
          error={error}
          onConnect={() => setAuth(true)}
          authRequired={authRequired}
        />
      </div>
      {auth && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={(e) => {
              e.preventDefault();
              sessionStorage.setItem("builder-token", token);
              refresh()
                .then(() => {
                  setAuth(false);
                  setError("");
                })
                .catch((err) => setError(err.message));
            }}
          >
            <h2>Connect to the quickstart server</h2>
            <p className="hint">
              Enter the access token from the server's .env (BUILDER_API_TOKEN).
            </p>
            {error && <p className="hint">{error}</p>}
            <label>
              Access token
              <input
                type="password"
                autoComplete="off"
                value={token}
                onChange={(e) => setToken(e.target.value)}
              />
            </label>
            <div className="actions">
              <button type="button" onClick={() => setAuth(false)}>
                Close
              </button>
              <button className="primary" type="submit">
                Connect
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
