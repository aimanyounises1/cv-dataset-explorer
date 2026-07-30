import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";

/** The zero-state's job.
 *
 * This used to offer five demo queries about dogs. A phrase to type teaches a
 * researcher nothing about their own corpus — it demonstrates that search
 * works, which they assumed. What they do not know on arrival is which of
 * these 8,000 images is worth opening.
 *
 * So the door states findings instead, and each one is a set you can open.
 * Both are already measured and already have endpoints — nothing new is
 * computed here, it is the same numbers /quality and /stats publish, moved to
 * where the first click happens.
 *
 * Each finding carries its own cut, because none of them is a fact: "suspect"
 * is a threshold on caption agreement and "contaminated" a threshold on a
 * cosine. A bare count would be a number whose robustness the reader cannot
 * check — the same argument LeakagePanel already makes for showing a curve
 * rather than a figure.
 */
const CONTAMINATION_CUT = 0.9;

/** Two measured findings, each opening the set it counts.
 *
 * Deliberately two, not four. A board of every signal the app computes would be
 * a feature list wearing a dashboard's clothes; these two are the ones with
 * consequences for a reported number — a held-out image that repeats a training
 * image, and a caption its own image does not support.
 *
 * Failure is silent by design: a finding that cannot be measured (no
 * embeddings, no QA scores) simply does not appear, because the alternative is
 * a zero that reads as "your dataset is clean".
 */
export default function FindingsRow() {
  const [leak, setLeak] = useState<{ n: number; of: number } | null>(null);
  const [ids, setIds] = useState<number[] | null>(null);
  const [qa, setQa] = useState<{ n: number; cut: number } | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    api.leakage(CONTAMINATION_CUT, ctrl.signal)
      .then((r) => setLeak({ n: r.contaminated, of: r.held_out_total }))
      .catch(() => { /* not measurable here — the card stays away */ });
    api.leakageContaminated(CONTAMINATION_CUT, ctrl.signal)
      .then((r) => setIds(r.ids))
      .catch(() => { /* the count can still show; the link is what needs ids */ });
    api.qaSummary()
      .then((s) => {
        if (!s.available) return;
        // The same cut /quality opens on: the value below which ~1% of captions
        // fall, read off this corpus rather than hard-coded.
        const total = s.histogram.reduce((n, b) => n + b.count, 0);
        let seen = 0;
        let cut = s.max_agreement ?? 1;
        for (const b of s.histogram) {
          seen += b.count;
          if (seen / total >= 0.01) { cut = b.hi; break; }
        }
        return api.qaSelection(cut, ctrl.signal)
          .then((sel) => setQa({ n: sel.samples, cut }));
      })
      .catch(() => { /* same contract as above */ });
    return () => ctrl.abort();
  }, []);

  const cards = [
    leak && ids && ids.length > 0 && {
      key: "leak",
      to: `/?ids=${ids.join(",")}`,
      n: leak.n,
      unit: `of ${leak.of.toLocaleString()} held-out images`,
      what: "have a near-twin in training",
      cut: `cosine ≥ ${CONTAMINATION_CUT.toFixed(2)}`,
      /* Not "partly memorisation" any more. That reading was borrowed from
         Barz & Denzler's CIFAR result, where the duplicated thing is the
         photograph; measured here with a perceptual hash, these pairs are no
         closer pixel-wise than two images picked at random from the corpus.
         `/api/stats/leakage/pixel` is where that measurement lives. */
      why: "The same subject, shot again — not the same photograph.",
    },
    qa && qa.n > 0 && {
      key: "qa",
      to: `/?max_agreement=${qa.cut}`,
      n: qa.n,
      unit: "images",
      what: "carry a caption the image does not support",
      cut: `agreement ≤ ${qa.cut.toFixed(3)}`,
      why: "Likely annotation errors, worst first.",
    },
  ].filter(Boolean) as {
    key: string; to: string; n: number; unit: string;
    what: string; cut: string; why: string;
  }[];

  if (!cards.length) return null;

  return (
    <section className="findings" aria-label="What this corpus gets wrong">
      <div className="eyebrow">Start from what is measured</div>
      <div className="findings-row">
        {cards.map((c) => (
          <Link key={c.key} className="finding" to={c.to}>
            <span className="finding-n">{c.n.toLocaleString()}</span>
            <span className="finding-unit">{c.unit}</span>
            <span className="finding-what">{c.what}</span>
            <span className="finding-cut">{c.cut}</span>
            <span className="finding-why">{c.why}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}
