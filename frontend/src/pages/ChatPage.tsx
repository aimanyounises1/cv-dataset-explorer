import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ChatMessage, ChatStatus, ChatTraceStep, SampleCard } from "../api/types";
import ImageCard from "../components/ImageCard";

interface Turn extends ChatMessage {
  samples?: SampleCard[];
  trace?: ChatTraceStep[];
}

const SUGGESTIONS = [
  "Show me dogs jumping into water",
  "What are the rarest slices of this dataset?",
  "Find the 5 most suspect captions and tag them as needs-review",
  "How is the dataset split, and is semantic search enabled?",
];

const STORAGE_KEY = "cvde-chat-turns";

function loadTurns(): Turn[] {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "[]") as Turn[];
  } catch {
    return [];
  }
}

/** Assistant view: a Fugu-style multi-agent orchestration (LangGraph + Ollama)
 * behind a single chat box. The trace chips show which specialist and tools
 * each answer came from. The conversation is kept in sessionStorage so
 * clicking through to a sample and coming back doesn't lose it. */
export default function ChatPage() {
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [turns, setTurns] = useState<Turn[]>(loadTurns);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    api.chatStatus().then(setStatus).catch(() =>
      setStatus({ available: false, model: "?", reason: "Backend unreachable." }));
  }, []);

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(turns));
    } catch {
      // Best effort: a full sessionStorage only loses persistence, not the chat.
    }
  }, [turns]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  const ask = async (text: string) => {
    const content = text.trim();
    if (!content || busy) return;
    const history: ChatMessage[] = [...turns.map(({ role, content }) => ({ role, content })),
                                    { role: "user", content }];
    setTurns((t) => [...t, { role: "user", content }]);
    setInput("");
    setBusy(true);
    try {
      const res = await api.chat(history);
      setTurns((t) => [...t, {
        role: "assistant", content: res.reply, samples: res.samples, trace: res.trace,
      }]);
    } catch (e) {
      setTurns((t) => [...t, {
        role: "assistant",
        content: `⚠️ ${e instanceof Error ? e.message : String(e)}`,
      }]);
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void ask(input);
  };

  if (status && !status.available) {
    return (
      <div className="panel" style={{ maxWidth: 720, margin: "40px auto" }}>
        <h3>Assistant unavailable</h3>
        <p style={{ color: "var(--text-dim)", lineHeight: 1.6 }}>{status.reason}</p>
        <p style={{ color: "var(--text-dim)", fontSize: 13 }}>
          The assistant is an optional layer — a Fugu-style multi-agent
          orchestration (LangGraph) running entirely locally via Ollama with
          model <code>{status.model}</code>. Everything else in the app works
          without it.
        </p>
        <button className="ghost" onClick={() => {
          setStatus(null);
          api.chatStatus().then(setStatus).catch(() => undefined);
        }}>
          Re-check
        </button>
      </div>
    );
  }

  return (
    <div className="chat-page">
      <div className="chat-scroll">
        {turns.length === 0 && (
          <div className="chat-empty">
            <div className="section-title" style={{ marginTop: 0 }}>Dataset assistant</div>
            <p className="meta-line" style={{ maxWidth: 640 }}>
              An orchestrator routes your request to specialist agents — retrieval
              (search, similar, inspect, tag) and insights (stats, coverage,
              caption QA) — then a quality gate verifies the answer. Runs fully
              locally{status ? ` on ${status.model}` : ""}.
            </p>
            <div className="chip-row">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chip" onClick={() => void ask(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={`chat-turn ${t.role}`}>
            <div className="chat-bubble">
              {t.trace && t.trace.length > 0 && (
                <div className="trace-row">
                  {t.trace.map((step, j) => (
                    <span key={j} className="trace-chip" title={step.input}>
                      {step.agent} → {step.tool}
                    </span>
                  ))}
                </div>
              )}
              <div className="chat-text">{t.content}</div>
              {t.samples && t.samples.length > 0 && (
                <div className="grid chat-grid">
                  {t.samples.map((s) => <ImageCard key={s.id} sample={s} />)}
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="chat-turn assistant">
            <div className="chat-bubble"><div className="chat-text loading-dots">
              Orchestrating agents…
            </div></div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <form className="chat-input" onSubmit={submit}>
        <input
          aria-label="Ask the dataset assistant"
          placeholder="Ask about the dataset, search for images, audit captions…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
        <button className="primary" type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
        {turns.length > 0 && (
          <button className="ghost" type="button" disabled={busy}
                  onClick={() => setTurns([])}>
            New chat
          </button>
        )}
      </form>
    </div>
  );
}
