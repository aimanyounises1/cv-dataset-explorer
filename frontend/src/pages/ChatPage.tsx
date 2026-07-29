import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Block } from "../api/blocks";
import type { ChatMessage, ChatStatus, ChatTraceStep, SampleCard } from "../api/types";
import BlockView from "../components/blocks";
import ImageCard from "../components/ImageCard";
import InlineMarkdown from "../components/InlineMarkdown";

interface Turn extends ChatMessage {
  /** What this entry actually is. A transport failure is not something the
   * assistant said, so it is marked and kept out of the history replayed to the
   * model. Absent on transcripts stored before the discriminator existed —
   * those are answers, which is the safe reading. */
  kind?: "answer" | "error";
  /** Set on a user turn while its answer is in flight, cleared when one lands.
   * The persistence effect writes it, so a session reopened after its run was
   * stopped — or after the tab left the page — says the question was
   * interrupted instead of ending on silence. */
  pending?: boolean;
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
interface LiveStep { node: string; label: string; t?: number }

/** A chat endpoint answering with a non-OK status. The status is carried as a
 * field rather than baked into the message so the 404 fallback can branch on it
 * while the message stays the server's own sentence — what reaches a reader
 * should read as a reason, not as a JSON fragment with a code in front. */
class ChatHttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "ChatHttpError";
  }
}

/** fetch() rejects an aborted request with a DOMException named AbortError.
 * Stopping is a decision, never a failure, so it is never reported as one. */
const isAbort = (e: unknown): boolean =>
  (e as { name?: string } | null | undefined)?.name === "AbortError";

/** Consume POST /api/chat/stream: NDJSON step lines (LangGraph's own node
 * transitions), then a final line carrying the exact blocking-endpoint
 * payload. Throws a 404 ChatHttpError when the endpoint is absent so the
 * caller can fall back to the blocking call. `signal` aborts the request and
 * the body read together, so Stop releases the connection rather than just
 * hiding its result. */
async function streamChat(
  history: ChatMessage[], onStep: (s: LiveStep) => void, signal: AbortSignal,
): Promise<import("../api/types").ChatResponse> {
  const r = await fetch("/api/chat/stream", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: history }),
  });
  if (!r.ok || !r.body) {
    let detail = "";
    try {
      const raw: unknown = (await r.json()).detail;
      detail = typeof raw === "string" ? raw : raw == null ? "" : JSON.stringify(raw);
    } catch {
      // Not a JSON body: the status line is all the server gave us.
    }
    throw new ChatHttpError(r.status, detail || `${r.status} ${r.statusText}`);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.type === "step") onStep(ev as LiveStep & { type: string });
      else if (ev.type === "final") return ev.response;
      else if (ev.type === "error") throw new Error(ev.detail);
    }
  }
  throw new Error("The stream ended without a final response.");
}

/** Task prompts, not tour prompts: each one exercises a real research
 * workflow the tools can actually complete. */
const SUGGESTIONS = [
  "Find rare night-time scenes and propose tagging the best candidates",
  "Which captions look wrong? Show the least-supported ones",
  "Inspect my latest album — what do its members share, and which are outliers?",
  "Show me dogs jumping into water",
  "Which time of day is hardest? Compare the slices",
];

/** The disabled opacity the control vocabulary already uses (index.css,
 * `button.ghost:disabled`), reused so a dimmed row of suggestions reads as the
 * same kind of "not right now" as a dimmed button rather than as a new state. */
const DISABLED_OPACITY = 0.4;

/** The gap `.notice` already puts between its stacked lines (index.css,
 * `.notice div + div`). The error block's action reuses it so the two failure
 * blocks — degraded and failed — space themselves the same way. */
const BLOCK_LINE_GAP = 6;

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

/** The first user turn, kept nearly whole. Renaming overwrites it for good.
 *
 * Stored at up to 200 characters rather than 60, because this string is the
 * only record of what the conversation was about and truncating at STORE time
 * destroys it: "Which captions in the test split disagree with…" and "Which
 * captions in the test split disagree with each other by more than 0.2?" are
 * the same 60 characters and different questions. Clamping is the LIST's job —
 * two lines of CSS, with the full text on the element's title and accessible
 * name — so what is shown can shrink without what is kept shrinking too. 200
 * bounds a pathological paste without touching a real question. */
const TITLE_MAX = 200;
const autoTitle = (text: string): string => {
  const t = text.trim().replace(/\s+/g, " ");
  return t.length <= TITLE_MAX ? t : `${t.slice(0, TITLE_MAX - 1).trimEnd()}…`;
};

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
  // Streamed graph steps for the in-flight turn; null when idle.
  const [liveSteps, setLiveSteps] = useState<LiveStep[] | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  // The in-flight turn's controller, so Stop, leaving the transcript and
  // leaving the page all have the same handle on it.
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    api.chatStatus().then(setStatus).catch(() =>
      setStatus({ available: false, model: "?", reason: "Backend unreachable." }));
  }, []);

  // No compute outlives the view that asked for it: unmounting stops the run
  // instead of leaving a graph burning a local GPU for an answer with nowhere
  // to land. The question keeps its `pending` marker, which is what tells the
  // reopened session it was interrupted.
  useEffect(() => () => abortRef.current?.abort(), []);

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
  // Waiting on the status probe is not the same as knowing it failed: the
  // composer stays live until the answer comes back false.
  const unavailable = status != null && !status.available;
  // Index of a question still marked in flight, or -1. Only the last turn can
  // carry the marker: it is set when a question is asked and cleared by
  // whatever lands after it.
  const interrupted =
    turns.length > 0 && turns[turns.length - 1].pending ? turns.length - 1 : -1;

  const openSession = (id: string) => {
    // Leaving a transcript ends its run rather than filing the reply under the
    // session you moved to. Nothing is silently lost: the question keeps its
    // `pending` marker and offers to be resent.
    abortRef.current?.abort();
    setActive(id);
    setTurns(loadSessionTurns(id));
    setConfirmDelete(null);
    setRenaming(null);
  };

  const startNew = () => {
    abortRef.current?.abort();
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
    // Deleting the transcript a run is writing into ends the run with it.
    if (active === id) abortRef.current?.abort();
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

  /** Clear the in-flight marker: the question has been answered, has failed, or
   * has been superseded, and in none of those cases is it still waiting. */
  const settled = (list: Turn[]): Turn[] =>
    list.map((t) => (t.pending ? { ...t, pending: undefined } : t));

  /** `base` is the transcript the question is asked against. A retry passes the
   * prefix before the exchange it is replacing, so re-asking replaces the failed
   * exchange instead of stacking a second copy of the question under the first. */
  const ask = async (text: string, base?: Turn[]) => {
    const content = text.trim();
    if (!content || busy || unavailable) return;
    const prior = base ?? turns;
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
    // A failed request is not prior assistant speech. Replaying one would
    // condition the model on a turn where it announced its own failure, which
    // is the single thing this app must never put in its mouth.
    const history: ChatMessage[] = [
      ...prior.filter((t) => t.kind !== "error")
              .map(({ role, content }) => ({ role, content })),
      { role: "user", content },
    ];
    setTurns([...prior, { role: "user", content, pending: true }]);
    setInput("");
    setBusy(true);
    setLiveSteps([]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      // The live trace is LangGraph's own update stream; a backend without
      // the stream endpoint (404) falls back to the blocking call.
      const res = await streamChat(history, (s) =>
        setLiveSteps((prev) => [...(prev ?? []), s]), ctrl.signal)
        .catch(async (e: unknown) => {
          if (e instanceof ChatHttpError && e.status === 404) {
            const blocking = await api.chat(history);
            // api.chat takes no signal, so on this path Stop drops the answer
            // rather than the request. Cancelling the request too needs
            // api/client.ts, which this view does not own.
            if (ctrl.signal.aborted) throw new DOMException("Stopped.", "AbortError");
            return blocking;
          }
          throw e;
        });
      setTurns((t) => [...settled(t), {
        role: "assistant", kind: "answer",
        content: res.reply, samples: res.samples, trace: res.trace,
        blocks: res.blocks, lanes: res.lanes, lanesFailed: res.lanes_failed,
        elapsed: res.elapsed_s,
      }]);
    } catch (e) {
      // Stopped on purpose: the question keeps its `pending` marker, and that
      // marker is what renders the interrupted notice and its Resend button.
      if (!isAbort(e)) {
        setTurns((t) => [...settled(t), {
          role: "assistant", kind: "error",
          content: e instanceof Error ? e.message : String(e),
        }]);
      }
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      const stamp = new Date().toISOString();
      setSessions((s) => s.map((m) => (m.id === id ? { ...m, updatedAt: stamp } : m)));
      setBusy(false);
      setLiveSteps(null);
    }
  };

  /** Re-ask the user turn at `at`, dropping it and everything after it. */
  const reask = (at: number) => {
    const q = turns[at];
    if (!q || q.role !== "user") return;
    void ask(q.content, turns.slice(0, at));
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void ask(input);
  };

  return (
    <div className="chat-page">
      {/* The page's one heading. Visually the canvas leads — the transcript
          state has no title bar to promote — so the h1 speaks only to
          assistive tech; the in-flow titles below stay non-headings. */}
      <h1 className="sr-only">Dataset assistant</h1>
      <aside className="chat-sessions" aria-label="Conversations">
        {/* The register stays navigable while a run is in flight — leaving a
            transcript stops its run rather than being forbidden by it. */}
        <button className="ghost chat-sessions-new" onClick={startNew}>
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
            const when = whenLabel(m.updatedAt);
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
                    {/* A title is the question that was asked, and questions are
                        long; the visible text is clamped to two lines. The whole
                        question stays reachable both ways — hover reads the
                        `title`, a screen reader reads the label — because a
                        register of past work you cannot identify is not a
                        register. */}
                    <button
                      className="chat-session-open"
                      onClick={() => openSession(m.id)}
                      title={m.title}
                      aria-label={when ? `Open "${m.title}" — ${when}` : `Open "${m.title}"`}
                      aria-current={isActive ? "true" : undefined}
                    >
                      <span className="chat-session-title">{m.title}</span>
                      <span className="chat-session-when">{when}</span>
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
              {/* Same contract as the register: the clamped line is what you
                  see, the full question is what you get on hover and what
                  assistive tech announces. */}
              <div className="chat-resume-list">
                {sorted.map((m) => {
                  const when = whenLabel(m.updatedAt);
                  return (
                    <button
                      key={m.id}
                      className="chat-resume"
                      onClick={() => openSession(m.id)}
                      title={m.title}
                      aria-label={when ? `Open "${m.title}" — ${when}` : `Open "${m.title}"`}
                    >
                      <span className="chat-resume-title">{m.title}</span>
                      <span className="chat-resume-when">{when}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
          {!landing && turns.length === 0 && (
            <div className="chat-empty">
              <div className="section-title" style={{ marginTop: 0 }}>Dataset assistant</div>
              <p className="meta-line" style={{ maxWidth: 660 }}>
                An orchestrator routes your request to specialist agents — up to two
                at once, in parallel — then a typed review step checks the response
                contract. This assistant is experimental: verify quantitative
                conclusions against its cited tool results and live charts. Charts
                and reports come back as components you can hover, sort and click
                through to the matching images, not as pictures. Runs fully locally
                {status ? ` on ${status.model}` : ""}.
              </p>
              {status?.specialists && status.specialists.length > 0 && (
                <p className="meta-line" style={{ maxWidth: 660 }}>
                  Specialists available: {status.specialists.join(", ")}. Ask{" "}
                  <em>how does this platform work architecturally?</em> to see the
                  topology drawn from the running registry.
                </p>
              )}
              {/* The suggestions stay legible with the model down — they are
                  the clearest statement of what the lanes can do — but they
                  are dimmed and inert, so nothing invites a click that would
                  do nothing. */}
              <div className="chip-row"
                   style={unavailable ? { opacity: DISABLED_OPACITY } : undefined}>
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="chip" disabled={unavailable}
                          onClick={() => void ask(s)}>{s}</button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            t.kind === "error" ? (
              /* Not a bubble and not the assistant's voice: a failed request
                 gets its own block, announced as an alert, so no reader — and
                 no later turn — mistakes it for something the model said. */
              <div key={i} className="error" role="alert">
                <div>The request failed. {t.content}</div>
                <button className="ghost" type="button" disabled={busy}
                        style={{ marginTop: BLOCK_LINE_GAP }}
                        onClick={() => reask(i - 1)}>
                  Retry
                </button>
              </div>
            ) : (
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
            )
          ))}
          {/* A question whose run was stopped — here or by leaving the page —
              says so and offers to be asked again, instead of leaving the
              transcript ending on a question nobody answered. */}
          {!busy && interrupted >= 0 && (
            <div className="notice" role="status">
              <div>
                This question was interrupted before an answer arrived — the run
                was stopped, or the page was left while it was still working.
              </div>
              <div>
                <button className="ghost" type="button" onClick={() => reask(interrupted)}>
                  Resend
                </button>
              </div>
            </div>
          )}
          {busy && (
            <div className="chat-turn assistant">
              <div className="chat-bubble">
                {/* The graph's own node transitions, streamed as they happen —
                    the last row is the step running right now. */}
                {liveSteps && liveSteps.length > 0 ? (
                  <ol className="live-trace" aria-live="polite">
                    {liveSteps.map((s, i) => (
                      <li key={i}
                          className={i === liveSteps.length - 1 ? "current" : "done"}>
                        <span className="lt-node">{s.node}</span>
                        <span className="lt-label">{s.label}</span>
                        {s.t != null && <span className="lt-t">{s.t}s</span>}
                      </li>
                    ))}
                  </ol>
                ) : (
                  <div className="chat-text loading-dots">Starting the graph…</div>
                )}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        {/* The model being down takes the composer away, not the register: the
            transcripts are localStorage and need nothing running to be read.
            The reason and its remedy stand where the input would be. */}
        {unavailable ? (
          <div className="chat-input">
            <div className="notice" role="status" style={{ flex: 1, marginBottom: 0 }}>
              <div><strong>Assistant unavailable.</strong> {status?.reason}</div>
              <div>
                The assistant is an optional layer — a Fugu-style multi-agent
                orchestration (LangGraph) running entirely locally via Ollama with
                model <code>{status?.model}</code>. Everything else in the app works
                without it, and the conversations above stay readable.
              </div>
              <div>
                <button className="ghost" type="button" onClick={() => {
                  setStatus(null);
                  api.chatStatus().then(setStatus).catch(() => undefined);
                }}>
                  Re-check
                </button>
              </div>
            </div>
          </div>
        ) : (
        <form className="chat-input" onSubmit={submit}>
          <input
            aria-label="Ask the dataset assistant"
            placeholder="Ask about the dataset, search for images, audit captions…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          {/* Send becomes Stop for the length of the run: a question that is
              taking two minutes and is going the wrong way can be taken back
              without abandoning the conversation to get out of it. */}
          {busy ? (
            <button className="ghost" type="button" onClick={() => abortRef.current?.abort()}>
              Stop
            </button>
          ) : (
            <button className="primary" type="submit" disabled={!input.trim()}>
              Send
            </button>
          )}
          {turns.length > 0 && (
            <button className="ghost" type="button" onClick={startNew}>
              New chat
            </button>
          )}
        </form>
        )}
      </div>
    </div>
  );
}
