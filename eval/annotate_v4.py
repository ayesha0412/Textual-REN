"""
Annotation tool for the v4 query results — visual labeling for presence
weight fitting.

What it does:
  1. Scans a run_dir for q*/result.json files
  2. Generates a self-contained annotation.html in that run_dir with:
     - One query card shown at a time (image + metadata + verdict buttons)
     - Keyboard shortcuts: 1=correct, 2=wrong, 3=partial, 4=skip,
                            5=false_negative, ←/→ to navigate, e=export
     - localStorage auto-save (won't lose work on accidental tab close)
     - Export button downloads verdicts to a JSON file
  3. Starts an HTTP server on localhost so the images load correctly
  4. Auto-opens your browser

Verdict vocabulary (binary mapping for the fit):
  correct           → 1  (system found the right object, or correctly
                          said not_found/uncertain for an absent object)
  wrong_object      → 0  (system found a *different* object)
  wrong_localization→ 0  (right object, wrong specific instance)
  partial           → 0  (right object, bad localization)
  false_negative    → 0  (system said not_found but object IS present)
  skip              → excluded from fit

After labeling: export the JSON, save it as eval/labels_<encoder>.json,
then run eval/fit_presence_weights.py.

Usage:
  python eval/annotate_v4.py --run-id 2026-06-17_siglip2_e01ce275
  python eval/annotate_v4.py --run-id 2026-06-14_358d195e --port 8766
"""

import argparse
import http.server
import json
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "eval"
RESULTS_DIR  = EVAL_DIR / "results"
QUERY_SET    = EVAL_DIR / "query_set_v4.yaml"


# ─── Data assembly ─────────────────────────────────────────────────────

def build_session_data(run_id: str) -> dict:
    """Walk the run_dir + query_set and produce the data the HTML needs."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        sys.exit(f"run dir not found: {run_dir}")

    qs = yaml.safe_load(QUERY_SET.read_text(encoding="utf-8"))
    qmap = {q["id"]: q for q in qs["queries"]}

    samples = []
    for qdir in sorted(run_dir.iterdir()):
        if not qdir.is_dir() or not qdir.name.startswith("q"):
            continue
        rj = qdir / "result.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        q = qmap.get(qdir.name, {})
        img = qdir / "debug_last_frame.jpg"
        samples.append({
            "qid": qdir.name,
            "video": q.get("video", ""),
            "query": r.get("query", q.get("text", "")),
            "expected": q.get("expected", "unknown"),
            "split": q.get("split", ""),
            "tags": q.get("tags", []),
            "decision": r.get("decision", "?"),
            "presence": r.get("presence", 0.0),
            "confidence": r.get("confidence", 0.0),
            "m_conf": r.get("evidence", {}).get("m_conf", 0.0),
            "refine_iters": r.get("refine_iters", 0),
            "image": f"./{qdir.name}/debug_last_frame.jpg" if img.exists() else None,
        })
    return {
        "run_id": run_id,
        "n_samples": len(samples),
        "samples": samples,
    }


# ─── HTML generation ───────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v4 Annotation — __RUN_ID__</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", sans-serif;
         background: #0f172a; color: #e2e8f0; }
  header { background: #1e293b; padding: 14px 24px;
           position: sticky; top: 0; z-index: 10;
           display: flex; gap: 24px; align-items: center;
           border-bottom: 1px solid #334155; }
  header h1 { font-size: 14px; font-weight: 600; color: #38bdf8; margin: 0; }
  .meta { color: #94a3b8; font-size: 13px; }
  .progress { flex: 1; background: #334155; height: 8px; border-radius: 4px;
              overflow: hidden; }
  .progress > div { height: 100%; background: #22c55e; transition: width .2s; }
  main { display: grid; grid-template-columns: 1fr 360px; gap: 24px;
         max-width: 1400px; margin: 24px auto; padding: 0 24px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
          overflow: hidden; }
  .card img { display: block; width: 100%; }
  .empty-img { padding: 60px; text-align: center; color: #64748b; }
  .side { display: flex; flex-direction: column; gap: 16px; }
  .meta-grid { background: #1e293b; border: 1px solid #334155;
               border-radius: 12px; padding: 16px; }
  .row { display: flex; justify-content: space-between; padding: 6px 0;
         border-bottom: 1px dashed #334155; font-size: 13px; }
  .row:last-child { border: none; }
  .row .k { color: #94a3b8; }
  .row .v { color: #e2e8f0; font-weight: 500; font-family: monospace; }
  .row .v.correct  { color: #4ade80; }
  .row .v.wrong    { color: #f87171; }
  .row .v.partial  { color: #fb923c; }
  .row .v.skip     { color: #94a3b8; }
  .verdict-buttons { display: grid; grid-template-columns: 1fr 1fr;
                     gap: 8px; }
  .verdict-buttons button { padding: 14px; border: 1px solid #334155;
                            background: #1e293b; color: #e2e8f0;
                            border-radius: 8px; font-size: 14px; font-weight: 500;
                            cursor: pointer; text-align: left; }
  .verdict-buttons button:hover  { border-color: #64748b; }
  .verdict-buttons .b1 { border-color: #166534; }
  .verdict-buttons .b1:hover, .verdict-buttons .b1.active { background: #166534; }
  .verdict-buttons .b2,.b3 { border-color: #b91c1c; }
  .verdict-buttons .b2:hover,.verdict-buttons .b2.active,
  .verdict-buttons .b3:hover,.verdict-buttons .b3.active { background: #b91c1c; }
  .verdict-buttons .b4 { border-color: #d97706; }
  .verdict-buttons .b4:hover, .verdict-buttons .b4.active { background: #d97706; }
  .verdict-buttons .b5 { border-color: #b91c1c; }
  .verdict-buttons .b5:hover, .verdict-buttons .b5.active { background: #b91c1c; }
  .verdict-buttons .b6 { border-color: #475569; }
  .verdict-buttons .b6:hover, .verdict-buttons .b6.active { background: #475569; }
  .verdict-buttons .key { color: #64748b; font-size: 11px; margin-left: 6px; }
  .nav { display: flex; justify-content: space-between; padding: 12px 0; }
  .nav button { padding: 10px 18px; background: #334155; color: #e2e8f0;
                border: none; border-radius: 6px; cursor: pointer;
                font-size: 13px; }
  .nav button:hover { background: #475569; }
  textarea { width: 100%; height: 60px; background: #0f172a; color: #e2e8f0;
             border: 1px solid #334155; border-radius: 6px; padding: 8px;
             font-family: inherit; font-size: 12px; resize: vertical; }
  #export { margin-top: 12px; width: 100%; padding: 14px;
            background: #38bdf8; color: #0f172a; border: none;
            border-radius: 8px; font-size: 14px; font-weight: 600;
            cursor: pointer; }
  #export:hover { background: #0ea5e9; }
  .tag { display: inline-block; padding: 2px 6px; background: #334155;
         border-radius: 4px; font-size: 10px; margin-right: 4px; }
  .summary { background: #0f172a; padding: 12px; border-radius: 6px;
             font-size: 12px; color: #94a3b8; }
  kbd { background: #334155; padding: 2px 6px; border-radius: 3px;
        font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>v4 Annotation</h1>
  <span class="meta" id="run-id">run: __RUN_ID__</span>
  <div class="progress"><div id="progress-fill"></div></div>
  <span class="meta" id="progress-text">0/0</span>
  <button id="export">Export labels.json</button>
</header>
<main>
  <div class="card" id="card">
    <div class="empty-img">loading…</div>
  </div>
  <aside class="side">
    <div class="meta-grid" id="meta"></div>
    <div class="verdict-buttons">
      <button class="b1" data-verdict="correct">
        <kbd>1</kbd> correct
        <span class="key">right object / right abstention</span>
      </button>
      <button class="b2" data-verdict="wrong_object">
        <kbd>2</kbd> wrong object
        <span class="key">box on different thing</span>
      </button>
      <button class="b3" data-verdict="wrong_localization">
        <kbd>3</kbd> wrong instance
        <span class="key">right kind, wrong one</span>
      </button>
      <button class="b4" data-verdict="partial">
        <kbd>4</kbd> partial
        <span class="key">right object, bad box</span>
      </button>
      <button class="b5" data-verdict="false_negative">
        <kbd>5</kbd> false neg
        <span class="key">said no but it's there</span>
      </button>
      <button class="b6" data-verdict="skip">
        <kbd>0</kbd> skip
        <span class="key">can't decide</span>
      </button>
    </div>
    <div class="nav">
      <button id="prev">← prev</button>
      <button id="next">next →</button>
    </div>
    <textarea id="note" placeholder="optional note…"></textarea>
    <div class="summary" id="summary"></div>
  </aside>
</main>
<script>
const DATA = __DATA__;
const STORAGE_KEY = "v4-labels-" + DATA.run_id;
let idx = 0;
let labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
  renderSummary();
}

function renderCard() {
  const s = DATA.samples[idx];
  const card = document.getElementById("card");
  card.innerHTML = s.image
    ? `<img src="${s.image}" alt="${s.qid}">`
    : `<div class="empty-img">(no debug image — system returned not_found before producing one)</div>`;

  const tags = s.tags.map(t => `<span class="tag">${t}</span>`).join("");
  const exp_color =
    s.expected === "present" ? "correct" :
    s.expected === "absent"  ? "wrong"   : "skip";
  document.getElementById("meta").innerHTML = `
    <div class="row"><span class="k">qid</span><span class="v">${s.qid}</span></div>
    <div class="row"><span class="k">video</span><span class="v">${s.video}</span></div>
    <div class="row"><span class="k">query</span><span class="v">"${s.query}"</span></div>
    <div class="row"><span class="k">expected</span><span class="v ${exp_color}">${s.expected}</span></div>
    <div class="row"><span class="k">split</span><span class="v">${s.split}</span></div>
    <div class="row"><span class="k">decision</span><span class="v">${s.decision}</span></div>
    <div class="row"><span class="k">presence</span><span class="v">${s.presence.toFixed(3)}</span></div>
    <div class="row"><span class="k">m_conf</span><span class="v">${s.m_conf.toFixed(4)}</span></div>
    <div class="row"><span class="k">refine_iters</span><span class="v">${s.refine_iters}</span></div>
    <div class="row"><span class="k">tags</span><span class="v">${tags}</span></div>
  `;

  // Highlight current verdict if labeled
  document.querySelectorAll(".verdict-buttons button").forEach(b => b.classList.remove("active"));
  const cur = labels[s.qid];
  if (cur) {
    const btn = document.querySelector(`[data-verdict="${cur.verdict}"]`);
    if (btn) btn.classList.add("active");
    document.getElementById("note").value = cur.note || "";
  } else {
    document.getElementById("note").value = "";
  }

  document.getElementById("progress-text").textContent =
    `${Object.keys(labels).length}/${DATA.n_samples} labeled · viewing ${idx + 1}`;
  document.getElementById("progress-fill").style.width =
    `${Object.keys(labels).length / DATA.n_samples * 100}%`;
}

function renderSummary() {
  const counts = {};
  Object.values(labels).forEach(l => { counts[l.verdict] = (counts[l.verdict] || 0) + 1; });
  document.getElementById("summary").innerHTML =
    "labeled by verdict: " +
    Object.entries(counts).map(([k,v]) => `<b>${k}</b>: ${v}`).join(" · ") +
    `<br>remaining: ${DATA.n_samples - Object.keys(labels).length}`;
}

function setVerdict(v) {
  const s = DATA.samples[idx];
  labels[s.qid] = {
    verdict: v,
    note: document.getElementById("note").value,
    presence: s.presence,
    decision: s.decision,
    expected: s.expected,
    timestamp: new Date().toISOString(),
  };
  save();
  renderCard();
  // Auto-advance if not the last
  if (idx < DATA.n_samples - 1) {
    idx++;
    renderCard();
  }
}

document.querySelectorAll(".verdict-buttons button").forEach(b => {
  b.addEventListener("click", () => setVerdict(b.dataset.verdict));
});
document.getElementById("note").addEventListener("blur", () => {
  const s = DATA.samples[idx];
  if (labels[s.qid]) {
    labels[s.qid].note = document.getElementById("note").value;
    save();
  }
});
document.getElementById("prev").addEventListener("click", () => {
  if (idx > 0) { idx--; renderCard(); }
});
document.getElementById("next").addEventListener("click", () => {
  if (idx < DATA.n_samples - 1) { idx++; renderCard(); }
});
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  const m = { "1":"correct","2":"wrong_object","3":"wrong_localization",
              "4":"partial","5":"false_negative","0":"skip" };
  if (m[e.key]) { setVerdict(m[e.key]); return; }
  if (e.key === "ArrowLeft"  && idx > 0) { idx--; renderCard(); }
  if (e.key === "ArrowRight" && idx < DATA.n_samples - 1) { idx++; renderCard(); }
  if (e.key === "e") { document.getElementById("export").click(); }
});
document.getElementById("export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({
    run_id: DATA.run_id,
    n_samples: DATA.n_samples,
    n_labeled: Object.keys(labels).length,
    exported_at: new Date().toISOString(),
    labels,
  }, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `labels_${DATA.run_id}.json`;
  a.click();
});

renderCard();
renderSummary();
</script>
</body>
</html>
"""


def generate_html(data: dict, output_path: Path) -> None:
    """Inline the session data into the HTML template."""
    html = (HTML_TEMPLATE
            .replace("__RUN_ID__", data["run_id"])
            .replace("__DATA__", json.dumps(data)))
    output_path.write_text(html, encoding="utf-8")


# ─── HTTP server ───────────────────────────────────────────────────────

def serve(run_dir: Path, port: int, open_browser: bool):
    import os
    os.chdir(str(run_dir))
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # silence
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/annotation.html"
        print(f"\n  Annotation tool: {url}")
        print(f"  Run dir         : {run_dir}")
        print(f"\n  Keyboard: 1=correct  2=wrong  3=wrong-instance  "
              f"4=partial  5=false-neg  0=skip  e=export  ←/→=navigate")
        print(f"\n  Ctrl-C to stop the server when done.\n")
        if open_browser:
            time.sleep(0.3)
            threading.Thread(target=lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True,
                    help="Subdir under eval/results/ to annotate.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open the browser.")
    args = p.parse_args()

    run_dir = RESULTS_DIR / args.run_id
    data = build_session_data(args.run_id)
    if data["n_samples"] == 0:
        sys.exit(f"No valid results in {run_dir}")

    output_path = run_dir / "annotation.html"
    generate_html(data, output_path)
    print(f"[annotate_v4] {data['n_samples']} samples loaded -> {output_path}")

    serve(run_dir, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
