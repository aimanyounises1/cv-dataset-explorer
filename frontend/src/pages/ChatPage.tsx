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

const STORAGE_KEY = "cvde-chat-turns";

function loadTurns(): Turn[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as Turn[];
  } catch {
    return [];
  }
}

/** Assistant view: a Fugu-style multi-agent orchestration (LangGraph + Ollama)
 * behind a single chat box. The trace chips show which specialist and tools
 * each answer came from. The conversation persists in localStorage — an
 * investigation's questions are work product, and closing the browser must
 * not erase them. "New chat" remains the explicit way to start over. */
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
      localStorage.setItem(STORAGE_KEY, JSON.stringify(turns));
    } catch {
      // Best effort: full storage only loses persistence, not the chat.
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
        blocks: res.blocks, lanes: res.lanes, lanesFailed: res.lanes_failed,
        elapsed: res.elapsed_s,
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
          <button className="ghost" type="button" disabled={busy}
                  onClick={() => setTurns([])}>
            New chat
          </button>
        )}
      </form>
    </div>
  );
}
