import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Block } from "../api/blocks";
import type { ChatMessage, ChatStatus, ChatTraceStep, SampleCard } from "../api/types";
import BlockView from "../components/blocks";
import ImageCard from "../components/ImageCard";
import InlineMarkdown from "../components/InlineMarkdown";

interface Turn extends ChatMessage {
  samples?: SampleCard[];
  trace?: ChatTraceStep[];
  blocks?: Block[];
  lanes?: string[];
  lanesFailed?: string[];
  elapsed?: number | null;
}

/** A conversation's registry entry. Turns live separately under
 * `cvde-chat-turns:<id>` so opening the list never parses every transcript. */
interface SessionMeta {
  id: string;
  title: string;
  updatedAt: string;
}

/** Deliberately spread across the four specialists, so the first thing a new
 * user clicks demonstrates a different lane each time — including the one that
 * reports on the application itself, which is otherwise undiscoverable. */
const SUGGESTIONS = [
  "Plot how the dataset splits into train, validation and test",
  "Which time of day is hardest? Compare the slices",
  "Show me dogs jumping into water",
  "Generate a dataset report",
  "How does this platform work architecturally?",
  "Show me the status of the application",
];

const REGISTRY_KEY = "cvde-chat-sessions";
/** The pre-sessions single-transcript key. It survives as the namespace prefix
 * so a stored transcript is recognisable in devtools as the same lineage. */
const LEGACY_KEY = "cvde-chat-turns";
const turnsKey = (id: string) => `${LEGACY_KEY}:${id}`;

const newId = (): string =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** One-time move of the single pre-sessions transcript into the registry.
 * Rollback-safe by ordering: the namespaced copy and the registry are written
 * BEFORE the legacy key is removed, so a crash mid-migration leaves the old
 * transcript where the next load can still find it. An unparseable legacy
 * value is never deleted — we only remove what we have provably copied. */
function migrateLegacy(): void {
  let legacy: string | null = null;
  try {
    if (localStorage.getItem(REGISTRY_KEY) != null) return;
    legacy = localStorage.getItem(LEGACY_KEY);
  } catch {
    return;
  }
  if (legacy == null) return;
  let turns: unknown;
  try {
    turns = JSON.parse(legacy);
  } catch {
    try {
      localStorage.setItem(REGISTRY_KEY, "[]");
    } catch {
      // Full storage: nothing to do; the app still opens with a fresh session.
    }
    return;
  }
  try {
    const registry: SessionMeta[] = [];
    if (Array.isArray(turns) && turns.length > 0) {
      const id = newId();
      localStorage.setItem(turnsKey(id), legacy);
      registry.push({
        id,
        title: "Earlier conversation",
        updatedAt: new Date().toISOString(),
      });
    }
    localStorage.setItem(REGISTRY_KEY, JSON.stringify(registry));
    localStorage.removeItem(LEGACY_KEY);
  } catch {
    // Full storage: leave everything in place; migration retries next load.
  }
}

function loadRegistry(): SessionMeta[] {
  migrateLegacy();
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(REGISTRY_KEY) ?? "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (m): m is SessionMeta =>
        !!m &&
        typeof m === "object" &&
        typeof (m as SessionMeta).id === "string" &&
        typeof (m as SessionMeta).title === "string" &&
        typeof (m as SessionMeta).updatedAt === "string",
    );
  } catch {
    return [];
  }
}

function loadSessionTurns(id: string): Turn[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(turnsKey(id)) ?? "[]");
    return Array.isArray(parsed) ? (parsed as Turn[]) : [];
  } catch {
    return [];
  }
}

/** ≤60 chars from the first user turn. Renaming overwrites it for good. */
const autoTitle = (text: string): string =>
  text.length <= 60 ? text : `${text.slice(0, 59).trimEnd()}…`;

function whenLabel(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toDateString() === new Date().toDateString()
    ? d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Assistant view: a Fugu-style multi-agent orchestration (LangGraph + Ollama)
 * behind a single chat box. The trace chips show which specialist and tools
 * each answer came from. Conversations persist in localStorage as named
 * sessions — an investigation's questions are work product, and neither
 * closing the browser nor starting a new chat may erase them. "New chat"
 * opens a fresh canvas; the previous conversation stays in the register. */
export default function ChatPage() {
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>(loadRegistry);
  // null = nothing open (the register leads); "new" = an explicit fresh canvas.
  const [active, setActive] = useState<string | "new" | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    api.chatStatus().then(setStatus).catch(() =>
      setStatus({ available: false, model: "?", reason: "Backend unreachable." }));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(REGISTRY_KEY, JSON.stringify(sessions));
    } catch {
      // Best effort: full storage only loses persistence, not the chat.
    }
  }, [sessions]);

  useEffect(() => {
    if (active === null || active === "new") return;
    try {
      localStorage.setItem(turnsKey(active), JSON.stringify(turns));
    } catch {
      // Best effort: full storage only loses persistence, not the chat.
    }
  }, [active, turns]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  // Newest first: the register is a recency list, like the library rail.
  const sorted = [...sessions].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  const landing = active === null && sessions.length > 0;

  const openSession = (id: string) => {
    // While a request is in flight the reply appends to the open transcript;
    // swapping transcripts mid-flight would file it under the wrong session.
    if (busy) return;
    setActive(id);
    setTurns(loadSessionTurns(id));
    setConfirmDelete(null);
    setRenaming(null);
  };

  const startNew = () => {
    if (busy) return;
    setActive("new");
    setTurns([]);
    setConfirmDelete(null);
    setRenaming(null);
  };

  const startRename = (m: SessionMeta) => {
    setRenaming(m.id);
    setDraft(m.title);
    setConfirmDelete(null);
  };

  const commitRename = () => {
    if (renaming === null) return;
    const title = draft.trim();
    if (title) {
      setSessions((s) => s.map((m) => (m.id === renaming ? { ...m, title } : m)));
    }
    setRenaming(null);
  };

  const deleteSession = (id: string) => {
    setSessions((s) => s.filter((m) => m.id !== id));
    try {
      localStorage.removeItem(turnsKey(id));
    } catch {
      // Best effort: an orphaned transcript key is invisible, not harmful.
    }
    setConfirmDelete(null);
    if (active === id) {
      setActive(null);
      setTurns([]);
    }
  };

  const ask = async (text: string) => {
    const content = text.trim();
    if (!content || busy) return;
    let id = active === null || active === "new" ? null : active;
    if (id === null) {
      // The session is born with its first turn, never empty: the register
      // lists conversations that happened, not canvases that were opened.
      id = newId();
      const meta: SessionMeta = {
        id, title: autoTitle(content), updatedAt: new Date().toISOString(),
      };
      setSessions((s) => [meta, ...s]);
      setActive(id);
    }
    const history: ChatMessage[] = [...turns.map(({ role, content }) => ({ role, content })),
                                    { role: "user", content }];
    setTurns((t) => [...t, { role: "user", content }]);
    setInput("");
    setBusy(true);
    try {
      const res = await api.chat(history);
      setTurns((t) => [...t, {
        role: "assistant", content: res.reply, samples: res.samples, trace: res.trace,
        blocks: res.blocks, lanes: res.lanes, lanesFailed: res.lanes_failed,
        elapsed: res.elapsed_s,
      }]);
    } catch (e) {
      setTurns((t) => [...t, {
        role: "assistant",
        content: `⚠️ ${e instanceof Error ? e.message : String(e)}`,
      }]);
    } finally {
      const stamp = new Date().toISOString();
      setSessions((s) => s.map((m) => (m.id === id ? { ...m, updatedAt: stamp } : m)));
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
        <h1 className="sr-only">Dataset assistant</h1>
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
      {/* The page's one heading. Visually the canvas leads — the transcript
          state has no title bar to promote — so the h1 speaks only to
          assistive tech; the in-flow titles below stay non-headings. */}
      <h1 className="sr-only">Dataset assistant</h1>
      <aside className="chat-sessions" aria-label="Conversations">
        <button className="ghost chat-sessions-new" onClick={startNew} disabled={busy}>
          New chat
        </button>
        {sorted.length === 0 && (
          <p className="chat-sessions-none">
            Conversations you start are kept here, on this machine.
          </p>
        )}
        <ul className="chat-session-list">
          {sorted.map((m) => {
            const isActive = m.id === active;
            return (
              <li key={m.id} className={`chat-session${isActive ? " active" : ""}`}>
                {renaming === m.id ? (
                  <input
                    className="chat-session-rename"
                    autoFocus
                    value={draft}
                    aria-label="Conversation title"
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={commitRename}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename();
                      else if (e.key === "Escape") setRenaming(null);
                    }}
                  />
                ) : (
                  <>
                    <button
                      className="chat-session-open"
                      onClick={() => openSession(m.id)}
                      disabled={busy}
                      title={m.title}
                      aria-current={isActive ? "true" : undefined}
                    >
                      <span className="chat-session-title">{m.title}</span>
                      <span className="chat-session-when">{whenLabel(m.updatedAt)}</span>
                    </button>
                    <span className="chat-session-acts">
                      <button
                        className="chat-session-act" title="Rename"
                        aria-label={`Rename "${m.title}"`}
                        onClick={() => startRename(m)}
                      >
                        ✎
                      </button>
                      {confirmDelete === m.id ? (
                        <button
                          className="chat-session-act danger"
                          aria-label={`Confirm deleting "${m.title}"`}
                          onClick={() => deleteSession(m.id)}
                        >
                          Delete?
                        </button>
                      ) : (
                        <button
                          className="chat-session-act" title="Delete"
                          aria-label={`Delete "${m.title}"`}
                          disabled={busy && isActive}
                          onClick={() => setConfirmDelete(m.id)}
                        >
                          ×
                        </button>
                      )}
                    </span>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      </aside>
      <div className="chat-main">
        <div className="chat-scroll">
          {landing && (
            <div className="chat-empty">
              <div className="section-title" style={{ marginTop: 0 }}>
                Pick up where you left off
              </div>
              <p className="meta-line" style={{ maxWidth: 660 }}>
                Your conversations are kept on this machine. Open one, or start
                typing below to begin a new one.
              </p>
              <div className="chat-resume-list">
                {sorted.map((m) => (
                  <button key={m.id} className="chat-resume" onClick={() => openSession(m.id)}>
                    <span className="chat-resume-title">{m.title}</span>
                    <span className="chat-resume-when">{whenLabel(m.updatedAt)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          {!landing && turns.length === 0 && (
            <div className="chat-empty">
              <div className="section-title" style={{ marginTop: 0 }}>Dataset assistant</div>
              <p className="meta-line" style={{ maxWidth: 660 }}>
                An orchestrator routes your request to specialist agents — up to two
                at once, in parallel — then a quality gate verifies the answer.
                Charts and reports come back as live components you can hover, sort
                and click through to the matching images, not as pictures. Runs
                fully locally{status ? ` on ${status.model}` : ""}.
              </p>
              {status?.specialists && status.specialists.length > 0 && (
                <p className="meta-line" style={{ maxWidth: 660 }}>
                  Specialists available: {status.specialists.join(", ")}. Ask{" "}
                  <em>how does this platform work architecturally?</em> to see the
                  topology drawn from the running registry.
                </p>
              )}
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
                {/* A lane that died is stated before the answer, not after it: the
                    reader needs to know the answer is partial while reading it. */}
                {t.lanesFailed && t.lanesFailed.length > 0 && (
                  <div className="notice" role="status">
                    {t.lanesFailed.join(" and ")}{" "}
                    {t.lanesFailed.length === 1 ? "did not finish" : "did not finish"} —
                    this answer covers only what the other specialists produced.
                  </div>
                )}
                <div className="chat-text">
                  {t.role === "assistant"
                    ? <InlineMarkdown text={t.content} />
                    : t.content}
                </div>
                {/* The canvas. Blocks render in the order the specialists produced
                    them, which is the order the request implied. */}
                {t.blocks && t.blocks.length > 0 && (
                  <div className="chat-blocks">
                    {t.blocks.map((b, k) => <BlockView key={k} block={b} />)}
                  </div>
                )}
                {t.samples && t.samples.length > 0 && (
                  <div className="grid chat-grid">
                    {t.samples.map((s) => <ImageCard key={s.id} sample={s} />)}
                  </div>
                )}
                {t.role === "assistant" && t.elapsed != null && (
                  <div className="chat-foot">
                    {t.lanes && t.lanes.length > 0
                      ? `${t.lanes.join(" ‖ ")} · ${t.elapsed.toFixed(1)}s`
                      : `${t.elapsed.toFixed(1)}s`}
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
            <button className="ghost" type="button" disabled={busy} onClick={startNew}>
              New chat
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
