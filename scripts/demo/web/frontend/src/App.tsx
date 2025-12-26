import React, { useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, defaultApiBase, withResultUrl } from "./api";
import { AskData, GroundData, GroundRecord, PhraseItem, SessionPayload } from "./types";

type AskMode = "default" | "roi" | "cot";

async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
      } else {
        reject(new Error("Failed to read file"));
      }
    };
    reader.onerror = () => reject(reader.error || new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

function deriveImageUrl(session: SessionPayload | null, apiBase: string, fallbackFile?: File): string | undefined {
  if (!session?.image) return fallbackFile ? URL.createObjectURL(fallbackFile) : undefined;
  const name = session.image.name;
  const dirUrl = session.session_dir_url;
  if (name && dirUrl) {
    return withResultUrl(apiBase, `${dirUrl}/${name}`);
  }
  const path = session.image.path;
  if (path && path.startsWith("/results")) {
    return withResultUrl(apiBase, path);
  }
  return fallbackFile ? URL.createObjectURL(fallbackFile) : undefined;
}

function App() {
  const [apiBase, setApiBase] = useState<string>(defaultApiBase());
  const [session, setSession] = useState<SessionPayload | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [question, setQuestion] = useState("What is in the image?");
  const [mode, setMode] = useState<AskMode>("default");
  const [modeIndex, setModeIndex] = useState<string>("");
  const [resetHistory, setResetHistory] = useState(false);
  const [answer, setAnswer] = useState<string>("");
  const [phrases, setPhrases] = useState<PhraseItem[]>([]);
  const [groundRecords, setGroundRecords] = useState<GroundRecord[]>([]);
  const [selectedPhrase, setSelectedPhrase] = useState<number | null>(null);
  const [imagePreview, setImagePreview] = useState<string | undefined>(undefined);
  const [status, setStatus] = useState<string>("Idle");
  const [loading, setLoading] = useState(false);
  const [askLoading, setAskLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    createSession(apiBase);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  const history = session?.history || [];

  const connectedText = useMemo(() => {
    if (!session) return "No session";
    return `Session ${session.session_id.slice(0, 8)} · last active ${new Date(session.last_active).toLocaleTimeString()}`;
  }, [session]);

  async function createSession(base: string) {
    setLoading(true);
    setStatus("Creating session...");
    try {
      const res = await apiFetch<SessionPayload>(base, "/session/create", { method: "POST", json: {} });
      setSession(res.data);
      setAnswer("");
      setPhrases([]);
      setGroundRecords([]);
      setSelectedPhrase(null);
      setImagePreview(undefined);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setStatus(res.message || "Session ready");
    } catch (err) {
      setStatus((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(fileList: FileList | null) {
    if (!session || !fileList?.length) return;
    const file = fileList[0];
    setUploading(true);
    setStatus("Uploading image...");
    try {
      const dataUrl = await fileToDataUrl(file);
      const res = await apiFetch<SessionPayload>(apiBase, "/load_image", {
        method: "POST",
        json: {
          session_id: session.session_id,
          image_base64: dataUrl,
        },
      });
      setSession(res.data);
      setImagePreview(deriveImageUrl(res.data, apiBase, file));
      setStatus(res.message || "Image loaded");
      if (fileInputRef.current) {
        // 清空 file input，便于重复选择同一张图也能触发 onChange
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setStatus((err as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk() {
    if (!session) {
      setStatus("No session");
      return;
    }
    if (!question.trim() && mode === "default") {
      setStatus("Question is required");
      return;
    }
    const indexForMode =
      mode === "default"
        ? undefined
        : modeIndex.trim()
          ? Number(modeIndex)
          : selectedPhrase !== null
            ? selectedPhrase
            : undefined;
    if (mode !== "default" && (indexForMode === undefined || Number.isNaN(indexForMode))) {
      setStatus("Index is required for ROI/CoT");
      return;
    }
    setAskLoading(true);
    setStatus("Asking...");
    try {
      const res = await apiFetch<AskData>(apiBase, "/ask", {
        method: "POST",
        json: {
          session_id: session.session_id,
          mode,
          question,
          index: indexForMode,
          reset_history: resetHistory,
        },
      });
      setAnswer(res.data.answer || "");
      setPhrases(res.data.phrases || []);
      if (res.data.history) {
        setSession((prev) => (prev ? { ...prev, history: res.data.history, history_turns: res.data.history_turns } : prev));
      }
      setStatus("Ask completed");
    } catch (err) {
      setStatus((err as Error).message);
    } finally {
      setAskLoading(false);
    }
  }

  async function handleGround() {
    if (!session) {
      setStatus("No session");
      return;
    }
    const index = selectedPhrase ?? (modeIndex ? Number(modeIndex) : undefined);
    if (index === null || index === undefined || Number.isNaN(index)) {
      setStatus("Select a phrase to ground");
      return;
    }
    setStatus("Grounding...");
    try {
      const res = await apiFetch<GroundData>(apiBase, "/ground", {
        method: "POST",
        json: { session_id: session.session_id, indices: [index] },
      });
      setGroundRecords(res.data.records || []);
      if (res.data.history) {
        setSession((prev) => (prev ? { ...prev, history: res.data.history } : prev));
      }
      setStatus("Ground completed");
    } catch (err) {
      setStatus((err as Error).message);
    }
  }

  async function handleClear() {
    if (!session) return;
    setStatus("Clearing history...");
    try {
      const res = await apiFetch<SessionPayload>(apiBase, "/clear", {
        method: "POST",
        json: { session_id: session.session_id },
      });
      setSession((prev) =>
        prev
          ? {
              ...prev,
              history: (res.data as any).history || [],
              history_turns: (res.data as any).history_turns,
            }
          : prev,
      );
      setAnswer("");
      setStatus("History cleared");
    } catch (err) {
      setStatus((err as Error).message);
    }
  }

  function renderHistory() {
    if (!history.length) {
      return <div className="placeholder">No messages yet. Ask something to start.</div>;
    }
    return history.map((item, idx) => (
      <div key={idx} className={`bubble ${item.role === "assistant" ? "assistant" : "user"}`}>
        <div className="bubble-role">{item.role || "unknown"}</div>
        <div className="bubble-text">{item.text}</div>
      </div>
    ));
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <div className="dot" />
          <div>
            <div className="title">F-LMM Web Demo</div>
            <div className="subtitle">{connectedText}</div>
          </div>
        </div>
        <div className="controls">
          <input
            className="input"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="API base, e.g. http://127.0.0.1:9000 or /api"
          />
          <button className="btn ghost" disabled={loading} onClick={() => createSession(apiBase)}>
            New session
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Chat</h2>
            <div className="status">{status}</div>
          </div>
          <div className="history">{renderHistory()}</div>

          <div className="form">
            <label className="label">Question</label>
            <textarea
              className="textarea"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              placeholder="Ask about the image..."
            />
            <div className="row">
              <div className="field">
                <label className="label">Mode</label>
                <select className="input" value={mode} onChange={(e) => setMode(e.target.value as AskMode)}>
                  <option value="default">default</option>
                  <option value="roi">roi</option>
                  <option value="cot">cot</option>
                </select>
              </div>
              {mode !== "default" && (
                <div className="field">
                  <label className="label">Index</label>
                  <input
                    className="input"
                    type="number"
                    value={modeIndex}
                    onChange={(e) => setModeIndex(e.target.value)}
                    placeholder="Phrase/ROI index"
                  />
                </div>
              )}
              <div className="field checkbox">
                <label className="label">
                  <input type="checkbox" checked={resetHistory} onChange={(e) => setResetHistory(e.target.checked)} /> reset history
                </label>
              </div>
            </div>
            <div className="row">
              <button className="btn primary" onClick={handleAsk} disabled={askLoading || loading}>
                {askLoading ? "Asking..." : "Send"}
              </button>
              <button className="btn ghost" onClick={handleClear} disabled={loading}>
                Clear history
              </button>
            </div>
            {answer && (
              <div className="answer">
                <div className="label">Answer</div>
                <div className="answer-body">{answer}</div>
              </div>
            )}
          </div>

          <div className="phrases">
            <div className="panel-subheader">
              <h3>Phrases</h3>
              <button className="btn ghost" onClick={handleGround} disabled={selectedPhrase === null}>
                Ground selected
              </button>
            </div>
            {phrases.length === 0 ? (
              <div className="placeholder">Ask first to get phrases.</div>
            ) : (
              <div className="phrase-list">
                {phrases.map((p) => (
                  <button
                    key={p.index}
                    className={`chip ${selectedPhrase === p.index ? "active" : ""}`}
                    onClick={() => setSelectedPhrase(p.index)}
                  >
                    #{p.index} {p.text}
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Image & Results</h2>
            <div className="status">{uploading ? "Uploading..." : ""}</div>
          </div>

          <div className="uploader">
            <label className="label">Upload image</label>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={(e) => handleUpload(e.target.files)} />
            <div className="hint">File is sent as base64 to backend /load_image.</div>
          </div>

          <div className="image-frame">
            {imagePreview ? (
              <img src={imagePreview} alt="current" />
            ) : (
              <div className="placeholder">No image loaded yet.</div>
            )}
          </div>

          <div className="ground">
            <div className="panel-subheader">
              <h3>Ground results</h3>
            </div>
            {groundRecords.length === 0 ? (
              <div className="placeholder">Select a phrase and run ground to view overlays.</div>
            ) : (
              <div className="ground-grid">
                {groundRecords.map((rec) => (
                  <div key={rec.index} className="ground-card">
                    <div className="ground-title">
                      #{rec.index} {rec.phrase}
                    </div>
                    <div className="thumbs">
                      {rec.overlay_url && <img src={withResultUrl(apiBase, rec.overlay_url)} alt="overlay" />}
                      {rec.mask_url && <img src={withResultUrl(apiBase, rec.mask_url)} alt="mask" />}
                      {rec.roi_url && <img src={withResultUrl(apiBase, rec.roi_url)} alt="roi" />}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
