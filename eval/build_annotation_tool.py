"""
Build a standalone HTML annotation tool.
Opens in browser — annotator clicks YES/NO for each query.
Results auto-save and compute all metrics.
"""

import json, os, base64, re
from pathlib import Path

ROOT = Path(r"D:\REN Project\REN")

# ------------------------------------------------------------------ #
# Collect all unique (query, video_id) pairs with their best image
# ------------------------------------------------------------------ #

ENTRIES = {}  # key=(video_id, query_clean) -> dict

def add(img_path, result_json_path, video_id, query_type="object"):
    img = Path(img_path)
    rj  = Path(result_json_path)
    if not img.exists() or not rj.exists():
        return
    with open(rj) as f:
        result = json.load(f)

    query = result.get("query", img.parent.name)
    key   = (video_id, query.lower().strip())

    # prefer annotate > vis > plain
    annotate = img.parent / "debug_last_frame_annotate.jpg"
    vis      = img.parent / "debug_last_frame_vis.jpg"
    best_img = annotate if annotate.exists() else (vis if vis.exists() else img)

    # skip if we already have a better source (eval/results/full preferred)
    existing = ENTRIES.get(key)
    if existing and "eval/results/full" in str(existing["img_path"]):
        return

    qt = "brand" if result.get("ocr_score", 0) > 0.5 else query_type
    ENTRIES[key] = {
        "video_id":       video_id,
        "query":          query,
        "query_type":     qt,
        "img_path":       str(best_img),
        "timestamp":      result.get("last_frame_timestamp"),
        "clip_sim":       result.get("clip_similarity", 0),
        "fused_sim":      result.get("fused_similarity", result.get("clip_similarity", 0)),
        "ocr_frames_hit": result.get("ocr_frames_hit", 0),
        "valid_segments": result.get("valid_segments"),
        "frames_above":   result.get("frames_above_threshold", 0),
        "pred_bbox":      result.get("pred_bbox"),
    }

# --- eval/results/full (most complete, authoritative) ---
full_dir = ROOT / "eval/results/full/P02_01"
if full_dir.exists():
    for folder in full_dir.iterdir():
        add(folder/"debug_last_frame.jpg", folder/"result.json", "P02_01")

# --- eval/evaluation_results ---
for folder in (ROOT/"eval/evaluation_results/P01_02").iterdir():
    add(folder/"debug_last_frame.jpg", folder/"result.json", "P01_02")

# --- eval/_tmp_eval ---
for folder in (ROOT/"eval/_tmp_eval/P01_01").iterdir():
    add(folder/"debug_last_frame.jpg", folder/"result.json", "P01_01")

# --- epic_results ---
vid_map = {"black dustbin":"P01_03","fork utensil":"P01_02","fork":"P01_02",
           "loaf of bread":"P01_05","strainer":"P01_05"}
for folder in (ROOT/"epic_results").iterdir():
    rj = folder/"result.json"
    if rj.exists():
        with open(rj) as f:
            q = json.load(f).get("query","").lower().strip()
        vid = vid_map.get(q, "P01")
        add(folder/"debug_last_frame.jpg", rj, vid)

# --- query_results/P02_01 (brand queries especially) ---
brand_qs = {"yorkshire_tea","fairy","twinings_camomile_tea","twinings_peppermint"}
for folder in (ROOT/"query_results/P02_01").iterdir():
    qt = "brand" if folder.name.lower() in brand_qs else "object"
    add(folder/"debug_last_frame.jpg", folder/"result.json", "P02_01", qt)

# --- query_results/P04_01_plate ---
add(ROOT/"query_results/P04_01_plate/debug_last_frame.jpg",
    ROOT/"query_results/P04_01_plate/result.json", "P04_01")

entries = sorted(ENTRIES.values(), key=lambda e: (e["query_type"], e["video_id"], e["query"]))
print(f"Total entries to annotate: {len(entries)}")
for e in entries:
    print(f"  [{e['query_type']:>8}] {e['video_id']} | {e['query']}")

# ------------------------------------------------------------------ #
# Encode images as base64
# ------------------------------------------------------------------ #

def img_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ------------------------------------------------------------------ #
# Build HTML
# ------------------------------------------------------------------ #

cards_js = []
for i, e in enumerate(entries):
    b64 = img_b64(e["img_path"])
    ts  = f"{e['timestamp']:.1f}s" if e['timestamp'] else "N/A"
    ocr = f"OCR hits: {e['ocr_frames_hit']}" if e['query_type'] == 'brand' else ""
    segs = f"{e['valid_segments']}" if e['valid_segments'] is not None else "N/A"
    cards_js.append({
        "id": i,
        "video": e["video_id"],
        "query": e["query"],
        "query_type": e["query_type"],
        "timestamp": ts,
        "clip_sim": round(e["clip_sim"], 3),
        "fused_sim": round(e["fused_sim"], 3),
        "ocr_note": ocr,
        "segments": segs,
        "frames_above": e["frames_above"],
        "img": b64,
    })

cards_json = json.dumps(cards_js, ensure_ascii=False)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Textual-REN Human Annotation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }}

  header {{
    background: #1e293b;
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid #334155;
    position: sticky; top: 0; z-index: 100;
  }}
  header h1 {{ font-size: 1.3rem; font-weight: 700; color: #38bdf8; }}
  #progress-bar-wrap {{ width: 320px; background: #334155; border-radius: 8px; height: 12px; }}
  #progress-bar {{ height: 12px; border-radius: 8px; background: #22c55e; width: 0%; transition: width .3s; }}
  #progress-text {{ font-size: .85rem; color: #94a3b8; margin-top: 4px; text-align: right; }}

  main {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px; }}

  .card {{
    background: #1e293b;
    border: 2px solid #334155;
    border-radius: 14px;
    margin-bottom: 28px;
    overflow: hidden;
    display: none;
  }}
  .card.active {{ display: flex; flex-direction: column; }}
  .card.done   {{ display: flex; opacity: .6; flex-direction: column; }}

  .card-header {{
    background: #0f172a;
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #334155;
  }}
  .query-badge {{
    font-size: 1.2rem;
    font-weight: 700;
    color: #f8fafc;
  }}
  .meta-pills {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .pill {{
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: .78rem;
    font-weight: 600;
    background: #334155;
    color: #cbd5e1;
  }}
  .pill.brand  {{ background: #7c3aed; color: #ede9fe; }}
  .pill.object {{ background: #0369a1; color: #bae6fd; }}
  .pill.ocr    {{ background: #15803d; color: #bbf7d0; }}

  .card-body {{
    display: flex;
    gap: 0;
  }}
  .img-wrap {{
    flex: 1;
    padding: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    min-height: 380px;
  }}
  .img-wrap img {{
    max-width: 100%;
    max-height: 480px;
    border-radius: 6px;
    object-fit: contain;
  }}
  .side-panel {{
    width: 260px;
    padding: 20px;
    border-left: 1px solid #334155;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }}
  .stat {{ font-size: .82rem; color: #94a3b8; }}
  .stat span {{ color: #e2e8f0; font-weight: 600; }}

  .question {{
    background: #0f172a;
    border-radius: 10px;
    padding: 14px;
    font-size: .95rem;
    color: #f1f5f9;
    border: 1px solid #334155;
    line-height: 1.5;
  }}
  .question strong {{ color: #38bdf8; }}

  .btn-row {{
    display: flex;
    gap: 10px;
    margin-top: auto;
  }}
  .btn {{
    flex: 1;
    padding: 14px;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform .1s, box-shadow .1s;
  }}
  .btn:active {{ transform: scale(.96); }}
  .btn-yes {{
    background: #16a34a;
    color: #fff;
    box-shadow: 0 0 0 0 #16a34a;
  }}
  .btn-yes:hover {{ background: #15803d; box-shadow: 0 0 12px #16a34a88; }}
  .btn-no {{
    background: #dc2626;
    color: #fff;
  }}
  .btn-no:hover {{ background: #b91c1c; box-shadow: 0 0 12px #dc262688; }}

  .answer-tag {{
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
  }}
  .answer-tag.yes {{ background: #14532d; color: #86efac; }}
  .answer-tag.no  {{ background: #450a0a; color: #fca5a5; }}

  #results-panel {{
    display: none;
    background: #1e293b;
    border-radius: 14px;
    padding: 28px;
    margin-top: 20px;
  }}
  #results-panel h2 {{ color: #38bdf8; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ color: #94a3b8; font-weight: 600; }}
  .big-metric {{ font-size: 2.4rem; font-weight: 800; color: #38bdf8; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .metric-box {{ background: #0f172a; border-radius: 10px; padding: 16px; border: 1px solid #334155; }}
  .metric-label {{ font-size: .8rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }}
  .metric-val   {{ font-size: 1.9rem; font-weight: 800; color: #f8fafc; }}
  .metric-val.green {{ color: #22c55e; }}
  .metric-val.orange {{ color: #f59e0b; }}
  .metric-val.red    {{ color: #ef4444; }}

  #download-btn {{
    margin-top: 20px;
    padding: 12px 28px;
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
  }}
  #download-btn:hover {{ background: #4f46e5; }}

  kbd {{
    background: #334155;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: .8rem;
    font-family: monospace;
  }}
  #shortcut-hint {{
    text-align: center;
    color: #64748b;
    font-size: .82rem;
    margin-bottom: 20px;
  }}
</style>
</head>
<body>

<header>
  <h1>🎯 Textual-REN — Human Precision Annotation</h1>
  <div>
    <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
    <div id="progress-text">0 / {len(entries)} annotated</div>
  </div>
</header>

<main>
  <div id="shortcut-hint">
    Press <kbd>Y</kbd> = Yes (correct) &nbsp;|&nbsp; <kbd>N</kbd> = No (incorrect) &nbsp;|&nbsp; <kbd>←</kbd> undo last
  </div>

  <div id="cards-container"></div>

  <div id="results-panel">
    <h2>📊 Annotation Results</h2>
    <div class="metric-grid" id="metric-grid"></div>
    <table id="results-table">
      <thead><tr><th>#</th><th>Query</th><th>Video</th><th>Type</th><th>CLIP Sim</th><th>Timestamp</th><th>Answer</th></tr></thead>
      <tbody id="results-body"></tbody>
    </table>
    <button id="download-btn" onclick="downloadResults()">⬇ Download JSON Results</button>
  </div>
</main>

<script>
const CARDS = {cards_json};
const answers = {{}};

function render() {{
  const container = document.getElementById('cards-container');
  container.innerHTML = '';
  CARDS.forEach((c, i) => {{
    const card = document.createElement('div');
    card.className = 'card' + (i === currentCard() ? ' active' : (answers[i] !== undefined ? ' done' : ''));
    card.id = 'card-' + i;

    const typeClass = c.query_type === 'brand' ? 'brand' : 'object';
    const ocrPill = c.ocr_note ? `<span class="pill ocr">🔍 ${{c.ocr_note}}</span>` : '';

    let answerHtml = '';
    if (answers[i] !== undefined) {{
      answerHtml = `<div class="answer-tag ${{answers[i] ? 'yes' : 'no'}}">${{answers[i] ? '✅ YES — Correct' : '❌ NO — Incorrect'}}</div>`;
    }} else {{
      answerHtml = `<div class="btn-row">
        <button class="btn btn-yes" onclick="annotate(${{i}}, true)">✅ YES — Correct</button>
        <button class="btn btn-no"  onclick="annotate(${{i}}, false)">❌ NO — Incorrect</button>
      </div>`;
    }}

    card.innerHTML = `
      <div class="card-header">
        <div>
          <div class="query-badge">Query ${{i+1}}/${{CARDS.length}}: "${{c.query}}"</div>
          <div style="color:#64748b;font-size:.8rem;margin-top:3px">Video: ${{c.video}}</div>
        </div>
        <div class="meta-pills">
          <span class="pill ${{typeClass}}">${{c.query_type.toUpperCase()}}</span>
          ${{ocrPill}}
          <span class="pill">CLIP: ${{c.clip_sim}}</span>
          <span class="pill">t=${{c.timestamp}}</span>
        </div>
      </div>
      <div class="card-body">
        <div class="img-wrap">
          <img src="data:image/jpeg;base64,${{c.img}}" alt="${{c.query}}" />
        </div>
        <div class="side-panel">
          <div class="question">
            <strong>Does the green bounding box correctly enclose "${{c.query}}"?</strong><br><br>
            ✅ <strong>YES</strong>: box tightly wraps the target object<br>
            ❌ <strong>NO</strong>: wrong object, wrong frame, or box is missing/misaligned
          </div>
          <div class="stat">Video: <span>${{c.video}}</span></div>
          <div class="stat">CLIP similarity: <span>${{c.clip_sim}}</span></div>
          <div class="stat">Fused similarity: <span>${{c.fused_sim}}</span></div>
          <div class="stat">Timestamp: <span>${{c.timestamp}}</span></div>
          <div class="stat">Temporal segments: <span>${{c.segments}}</span></div>
          <div class="stat">Frames above τ: <span>${{c.frames_above}}</span></div>
          ${{c.ocr_note ? `<div class="stat">OCR: <span style="color:#86efac">${{c.ocr_note}}</span></div>` : ''}}
          ${{answerHtml}}
        </div>
      </div>`;
    container.appendChild(card);
  }});
  updateProgress();
}}

function currentCard() {{
  for (let i = 0; i < CARDS.length; i++) {{
    if (answers[i] === undefined) return i;
  }}
  return -1;
}}

function annotate(i, yes) {{
  answers[i] = yes;
  const next = currentCard();
  render();
  if (next === -1) showResults();
  else {{
    const el = document.getElementById('card-' + next);
    if (el) el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
  }}
}}

function updateProgress() {{
  const done = Object.keys(answers).length;
  const pct  = done / CARDS.length * 100;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = done + ' / ' + CARDS.length + ' annotated';
}}

document.addEventListener('keydown', e => {{
  const cur = currentCard();
  if (e.key === 'y' || e.key === 'Y') {{ if (cur >= 0) annotate(cur, true); }}
  if (e.key === 'n' || e.key === 'N') {{ if (cur >= 0) annotate(cur, false); }}
  if (e.key === 'ArrowLeft') {{
    // undo last
    const done = Object.keys(answers).map(Number).sort((a,b)=>b-a);
    if (done.length > 0) {{ delete answers[done[0]]; render(); }}
  }}
}});

function showResults() {{
  document.getElementById('results-panel').style.display = 'block';
  document.getElementById('results-panel').scrollIntoView({{behavior:'smooth'}});

  const byType = {{}};
  let totalYes = 0, total = 0;
  CARDS.forEach((c, i) => {{
    if (answers[i] === undefined) return;
    const yes = answers[i] ? 1 : 0;
    totalYes += yes; total++;
    if (!byType[c.query_type]) byType[c.query_type] = {{yes:0, n:0}};
    byType[c.query_type].yes += yes;
    byType[c.query_type].n++;
  }});

  const overallPrec = total > 0 ? (totalYes / total * 100) : 0;

  let grid = `<div class="metric-box">
    <div class="metric-label">Overall Precision</div>
    <div class="metric-val ${{overallPrec>=70?'green':overallPrec>=50?'orange':'red'}}">${{overallPrec.toFixed(0)}}%</div>
    <div style="font-size:.8rem;color:#64748b">(${{totalYes}}/${{total}} correct)</div>
  </div>`;
  Object.entries(byType).forEach(([qt, s]) => {{
    const p = s.n > 0 ? (s.yes/s.n*100) : 0;
    grid += `<div class="metric-box">
      <div class="metric-label">Precision — ${{qt}}</div>
      <div class="metric-val ${{p>=70?'green':p>=50?'orange':'red'}}">${{p.toFixed(0)}}%</div>
      <div style="font-size:.8rem;color:#64748b">(${{s.yes}}/${{s.n}} correct)</div>
    </div>`;
  }});
  document.getElementById('metric-grid').innerHTML = grid;

  const tbody = document.getElementById('results-body');
  tbody.innerHTML = '';
  CARDS.forEach((c, i) => {{
    if (answers[i] === undefined) return;
    const tr = document.createElement('tr');
    const ans = answers[i];
    tr.innerHTML = `<td>${{i+1}}</td><td>${{c.query}}</td><td>${{c.video}}</td>
      <td><span class="pill ${{c.query_type}}">${{c.query_type}}</span></td>
      <td>${{c.clip_sim}}</td><td>${{c.timestamp}}</td>
      <td style="color:${{ans?'#22c55e':'#ef4444'}};font-weight:700">${{ans?'YES':'NO'}}</td>`;
    tbody.appendChild(tr);
  }});
}}

function downloadResults() {{
  const results = CARDS.map((c, i) => ({{
    id: i, query: c.query, video: c.video, query_type: c.query_type,
    clip_similarity: c.clip_sim, timestamp: c.timestamp,
    correct: answers[i] !== undefined ? answers[i] : null
  }}));

  const byType = {{}};
  let totalYes=0, total=0;
  results.forEach(r => {{
    if (r.correct === null) return;
    const y = r.correct ? 1 : 0;
    totalYes += y; total++;
    if (!byType[r.query_type]) byType[r.query_type] = {{yes:0,n:0}};
    byType[r.query_type].yes += y;
    byType[r.query_type].n++;
  }});

  const metrics = {{
    overall_precision: total>0 ? totalYes/total : null,
    total_annotated: total,
    total_correct: totalYes,
  }};
  Object.entries(byType).forEach(([qt,s]) => {{
    metrics['precision_' + qt] = s.n>0 ? s.yes/s.n : null;
    metrics['n_' + qt] = s.n;
    metrics['correct_' + qt] = s.yes;
  }});

  const blob = new Blob([JSON.stringify({{metrics, per_query: results}}, null, 2)],
                        {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'textual_ren_annotation_results.json';
  a.click();
}}

render();
</script>
</body>
</html>"""

out = ROOT / "eval" / "annotation_tool.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\n{'='*60}")
print(f"  Annotation tool: {out}")
print(f"  Total items:     {len(entries)}")
print(f"{'='*60}")
print("\nOpen this file in your browser and annotate all queries.")
print("Use Y / N keys or click buttons. Download JSON when done.")
