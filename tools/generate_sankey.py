#!/usr/bin/env python3
"""Generate an interactive Sankey diagram from a taxonomy-mapping CSV.

Reads a CSV whose columns describe a multi-stage mapping — e.g. the planktonzilla
taxonomy crosswalk that routes each source ``Dataset``/``Raw_Labels`` row through the
Linnaean ranks (``Kingdom -> Phylum -> ... -> Species``) down to a unified
``proposed_label`` — and emits a single self-contained HTML file. The page uses
Plotly via CDN, so there is no build step and no third-party Python dependency
(stdlib only). In the browser each column is a toggleable, drag-to-reorder stage
and flow width equals the number of label rows.

Examples
--------
    # Defaults: read the bundled taxonomy CSV, write planktonzilla_taxonomy_sankey.html
    python tools/generate_sankey.py

    # Custom input/output and open it when done
    python tools/generate_sankey.py --csv data.csv --out flow.html --open

    # Pick the stages shown on first load
    python tools/generate_sankey.py --stages Dataset,Kingdom,Phylum,Class,Order

    # Expose every column in the CSV as a selectable stage
    python tools/generate_sankey.py --all-columns
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import webbrowser
from pathlib import Path

# Curated stage order + friendly labels for the planktonzilla taxonomy crosswalk.
# Columns not listed here still work via --all-columns / --columns (label = column name).
PREFERRED: list[tuple[str, str]] = [
    ("Dataset", "Source dataset"),
    ("root_class", "Root class"),
    ("plankton", "Plankton?"),
    ("living", "Living?"),
    ("Kingdom", "Kingdom"),
    ("Phylum", "Phylum"),
    ("Class", "Class"),
    ("Order", "Order"),
    ("Family", "Family"),
    ("Genus", "Genus"),
    ("Species", "Species"),
    ("qualifier", "Qualifier"),
    ("proposed_label", "Proposed label"),
]

DEFAULT_STAGES = ["Dataset", "Kingdom", "Phylum", "Class"]
DEFAULT_TITLE = "planktonzilla taxonomy — interactive Sankey"


def default_csv() -> Path:
    """Locate the bundled taxonomy CSV relative to this script (repo/tools/)."""
    repo = Path(__file__).resolve().parent.parent
    return repo / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"


def resolve_dims(header: list[str], columns: str | None, all_columns: bool) -> list[dict[str, str]]:
    """Decide which columns become selectable stages, in display order."""
    labels = dict(PREFERRED)
    if columns:
        wanted = [c.strip() for c in columns.split(",") if c.strip()]
    elif all_columns:
        wanted = list(header)
    else:
        # Curated columns that exist in this CSV, in the preferred order.
        wanted = [k for k, _ in PREFERRED if k in header]
        if not wanted:  # unfamiliar CSV: fall back to every column
            wanted = list(header)
    missing = [c for c in wanted if c not in header]
    if missing:
        raise SystemExit(f"error: column(s) {missing} not in CSV header {header}")
    return [{"key": k, "label": labels.get(k, k)} for k in wanted]


def build_html(csv_path: Path, dims: list[dict[str, str]], stages: list[str], title: str) -> str:
    keys = [d["key"] for d in dims]
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Always emit root_class per record (Improvement 3 — lineage coloring), even when it
    # is not a selected stage. Guard for CSVs lacking the column: if root_class is absent
    # from the source rows it is simply omitted and the JS tolerates a missing/empty value.
    has_root_class = bool(rows) and "root_class" in rows[0]
    rec_keys = list(keys)
    if has_root_class and "root_class" not in rec_keys:
        rec_keys.append("root_class")
    records = [{k: (r.get(k) or "").strip() for k in rec_keys} for r in rows]
    payload = json.dumps({"dims": dims, "records": records}, separators=(",", ":"))
    # Defang any literal </script> in data so it can't break out of the JSON island.
    payload = payload.replace("</", "<\\/")
    return (
        TEMPLATE.replace("__TITLE__", title)
        .replace("__PAYLOAD__", payload)
        .replace("__DEFAULT__", json.dumps(stages))
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="generate_sankey.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=default_csv(), help="input CSV (default: bundled taxonomy CSV)")
    ap.add_argument("--out", type=Path, default=Path("planktonzilla_taxonomy_sankey.html"), help="output HTML path")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES), help="comma-separated stages shown on first load")
    ap.add_argument("--columns", default=None, help="comma-separated columns to expose as selectable stages")
    ap.add_argument("--all-columns", action="store_true", help="expose every CSV column as a stage")
    ap.add_argument("--title", default=DEFAULT_TITLE, help="page title / heading")
    ap.add_argument("--open", dest="open_browser", action="store_true", help="open the result in a browser when done")
    args = ap.parse_args(argv)

    if not args.csv.exists():
        raise SystemExit(f"error: CSV not found: {args.csv}")
    with args.csv.open(newline="") as fh:
        header = next(csv.reader(fh), [])
    if not header:
        raise SystemExit(f"error: CSV is empty: {args.csv}")

    dims = resolve_dims(header, args.columns, args.all_columns)
    dim_keys = {d["key"] for d in dims}
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in dim_keys]
    if bad:
        raise SystemExit(f"error: --stages {bad} not among selectable columns {sorted(dim_keys)}")

    html = build_html(args.csv, dims, stages, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    with args.csv.open(newline="") as fh:
        n_rows = sum(1 for _ in csv.DictReader(fh))
    print(f"wrote {args.out} ({len(html):,} bytes)")
    print(f"  rows={n_rows}  stages={stages}  selectable_columns={[d['key'] for d in dims]}")
    if args.open_browser:
        webbrowser.open(args.out.resolve().as_uri())
        print(f"  opened {args.out.resolve().as_uri()}")
    return 0


# --------------------------------------------------------------------------- #
# Self-contained HTML template. Placeholders: __TITLE__ __PAYLOAD__ __DEFAULT__
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inria+Sans:wght@400;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Inria+Serif:wght@400;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{ --bg:#ffffff; --panel:#f5f6f7; --ink:#1a1a1a; --mut:#6b7280; --acc:#c9191e; --line:#e3e6ea; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Inria Sans",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
  header{padding:16px 24px;border-top:3px solid var(--acc);border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:4px}
  h1{font-family:"Inria Serif",Georgia,"Times New Roman",serif;font-size:20px;margin:0;font-weight:700;color:var(--ink)}
  .sub{color:var(--mut);font-size:12px}
  .wrap{display:flex;height:calc(100vh - 72px)}
  aside{width:300px;flex:0 0 300px;background:var(--panel);border-right:1px solid var(--line);padding:18px;overflow:auto}
  main{flex:1;min-width:0;padding:8px 12px;overflow:auto}
  .grp{margin-bottom:20px}
  .grp h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 8px;font-weight:700}
  .dim{display:flex;align-items:center;gap:8px;padding:7px 9px;border:1px solid var(--line);border-radius:0;margin-bottom:6px;background:#fff;cursor:grab;user-select:none}
  .dim.off{opacity:.5}
  .dim .order{width:20px;height:20px;border-radius:50%;background:#e9ecef;color:var(--mut);font-size:11px;display:grid;place-items:center;flex:0 0 auto}
  .dim.on .order{background:var(--acc);color:#fff;font-weight:700}
  .dim .name{flex:1}
  .dim .nm-sub{color:var(--mut);font-size:11px}
  .dim input{accent-color:var(--acc)}
  label.row{display:flex;align-items:center;gap:8px;margin:8px 0;color:var(--ink)}
  input[type=range]{width:100%;accent-color:var(--acc)}
  .btn{display:block;width:100%;text-align:left;padding:8px 12px;border-radius:0;border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer;font:inherit;font-size:12px;margin-bottom:6px}
  .btn:hover{border-color:var(--acc);color:var(--acc)}
  .hint{color:var(--mut);font-size:11px;margin-top:4px}
  #chart{width:100%;min-height:600px}
  .pill{display:inline-block;background:#fbe9ea;padding:1px 8px;color:var(--acc);font-weight:700;font-variant-numeric:tabular-nums}
  code{background:#f0f1f3;padding:1px 5px;color:#b3121a}
  .lgd{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px;color:var(--ink)}
  .lgd .sw{width:14px;height:14px;flex:0 0 auto;border:1px solid rgba(0,0,0,.15)}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub"><span id="meta"></span> &middot; flow width = number of label rows</div>
</header>
<div class="wrap">
  <aside>
    <div class="grp">
      <h2>Flow stages (drag to reorder)</h2>
      <div id="dims"></div>
      <div class="hint">Toggle columns on/off; drag to set left&rarr;right order. Each enabled column is a stage.</div>
    </div>
    <div class="grp">
      <h2>Options</h2>
      <label class="row"><input type="checkbox" id="hideEmpty"> Drop rows blank in any stage</label>
      <label class="row">Aggregate below: <span class="pill" id="minLbl">1</span> taxa</label>
      <input type="range" id="minFlow" min="1" max="40" value="1"/>
      <div class="hint">Collapse nodes smaller than N into a muted &ldquo;Other (n taxa)&rdquo;
        node per stage (flow is conserved, nothing is dropped).</div>
    </div>
    <div class="grp">
      <h2>Lineage (root class)</h2>
      <div id="legend"></div>
      <div class="hint">Each link is colored by the majority root&nbsp;class of the records flowing through it.</div>
    </div>
    <div class="grp">
      <h2>Presets</h2>
      <button class="btn" data-preset="Dataset,Kingdom,Phylum,Class">Source &rarr; taxonomy</button>
      <button class="btn" data-preset="Kingdom,Phylum,Class,Order,Family">Linnaean ranks</button>
      <button class="btn" data-preset="Dataset,root_class,plankton,living">Source &rarr; category</button>
      <button class="btn" data-preset="Class,Order,Family,Genus,proposed_label">Lower ranks &rarr; label</button>
    </div>
  </aside>
  <main><div id="chart"></div></main>
</div>

<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const DIMS = DATA.dims;
const RECORDS = DATA.records;
const LABELS = Object.fromEntries(DIMS.map(d=>[d.key,d.label]));
const AVAIL = new Set(DIMS.map(d=>d.key));
let selected = (__DEFAULT__).filter(k=>AVAIL.has(k));

document.getElementById('meta').textContent =
  RECORDS.length.toLocaleString() + " label rows · " + DIMS.length + " columns";

const STAGE_COLORS = ["#c9191e","#2b5c8a","#3f8e6b","#c77f2e","#6b5b95","#3f8e9e","#a23b72","#7a8290","#b0563f"];
function hexA(h,a){const n=parseInt(h.slice(1),16);return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;}
const EMPTY = "(blank)";
const MUTED = "#aab3bf";          // shared muted gray for (blank) and Other nodes
const OTHER_PREFIX = "Other (";   // Other-node label prefix; full label = "Other (n taxa)"

// Categorical palette for lineage (root_class) link coloring. Derived at runtime from the
// distinct root_class values in the data (root_class has <=4 distinct values in this CSV);
// "Unknown" (empty/missing root_class) always maps to the neutral muted gray.
const LINEAGE_PALETTE = ["#c9191e","#2b5c8a","#3f8e6b","#c77f2e","#6b5b95","#a23b72"];
const LINEAGE_UNKNOWN = "Unknown";
function buildLineageColors(){
  const seen=[];
  for(const rec of RECORDS){ const rc=(rec.root_class||"").trim(); const key=rc===""?LINEAGE_UNKNOWN:rc;
    if(!seen.includes(key)) seen.push(key); }
  // Stable order: known values first (in first-seen order), Unknown last.
  const known=seen.filter(k=>k!==LINEAGE_UNKNOWN).sort();
  const order=[...known]; if(seen.includes(LINEAGE_UNKNOWN)) order.push(LINEAGE_UNKNOWN);
  const map=new Map(); let i=0;
  for(const k of order){ map.set(k, k===LINEAGE_UNKNOWN ? MUTED : LINEAGE_PALETTE[i++ % LINEAGE_PALETTE.length]); }
  return map;
}
const LINEAGE_COLORS = buildLineageColors();
function renderLegend(){
  const box=document.getElementById('legend'); if(!box) return;
  box.innerHTML='';
  for(const [k,c] of LINEAGE_COLORS){
    const el=document.createElement('div'); el.className='lgd';
    el.innerHTML=`<span class="sw" style="background:${c}"></span><span>${k}</span>`;
    box.appendChild(el);
  }
}

const dimsBox = document.getElementById('dims');
function renderDims(){
  const rest = DIMS.map(d=>d.key).filter(k=>!selected.includes(k));
  const order = [...selected, ...rest];
  dimsBox.innerHTML='';
  order.forEach((key)=>{
    const d = DIMS.find(x=>x.key===key);
    const on = selected.includes(key);
    const idx = selected.indexOf(key);
    const el = document.createElement('div');
    el.className = 'dim '+(on?'on':'off');
    el.draggable = true; el.dataset.key = key;
    el.innerHTML = `<span class="order">${on?idx+1:'·'}</span>
      <input type="checkbox" ${on?'checked':''}/>
      <span class="name">${d.label}<div class="nm-sub">${key}</div></span>`;
    el.querySelector('input').addEventListener('change',(e)=>{
      e.stopPropagation();
      if(e.target.checked){ if(!selected.includes(key)) selected.push(key); }
      else { selected = selected.filter(k=>k!==key); }
      renderDims(); draw();
    });
    el.addEventListener('dragstart',ev=>{ev.dataTransfer.setData('text/plain',key); el.style.opacity=.4;});
    el.addEventListener('dragend',()=>{el.style.opacity='';});
    el.addEventListener('dragover',ev=>ev.preventDefault());
    el.addEventListener('drop',ev=>{
      ev.preventDefault();
      const src = ev.dataTransfer.getData('text/plain');
      if(src===key) return;
      if(!selected.includes(src)) selected.push(src);
      if(!selected.includes(key)) selected.push(key);
      selected = selected.filter(k=>k!==src);
      const at = selected.indexOf(key);
      selected.splice(at,0,src);
      renderDims(); draw();
    });
    dimsBox.appendChild(el);
  });
}

function argmaxKey(m){ let bk=null,bv=-1; for(const [k,v] of m){ if(v>bv){bv=v;bk=k;} } return bk; }
function mergeTally(dst,src){ for(const [k,v] of src){ dst.set(k,(dst.get(k)||0)+v); } }

function build(){
  const hideEmpty = document.getElementById('hideEmpty').checked;
  const minFlow = +document.getElementById('minFlow').value;  // aggregation threshold (taxa)
  const stages = selected;
  if(stages.length<2) return {empty:true};

  // -- Pass 1: count raw per-(stage,label) nodes and raw links, tallying root_class per link. --
  const rawIndex = new Map(); const rawNodes=[];   // rawNodes[i] = {stage, label}
  function rnid(si,val){
    const k=si+" "+val;
    if(!rawIndex.has(k)){ rawIndex.set(k,rawNodes.length); rawNodes.push({stage:si,label:val}); }
    return rawIndex.get(k);
  }
  const linkMap=new Map();      // "s>t" -> count
  const linkLineage=new Map();  // "s>t" -> Map(root_class -> count)
  let used=0;
  for(const rec of RECORDS){
    const vals = stages.map(k=>{const v=rec[k]; return v===''?EMPTY:v;});
    if(hideEmpty && vals.includes(EMPTY)) continue;
    used++;
    const rc=(rec.root_class||"").trim() || LINEAGE_UNKNOWN;
    for(let i=0;i<stages.length-1;i++){
      const s=rnid(i,vals[i]); const t=rnid(i+1,vals[i+1]);
      const kk=s+">"+t;
      linkMap.set(kk,(linkMap.get(kk)||0)+1);
      let tal=linkLineage.get(kk); if(!tal){tal=new Map();linkLineage.set(kk,tal);}
      tal.set(rc,(tal.get(rc)||0)+1);
    }
  }

  // -- Per-node total flow = sum of incident link values for that node's stage. --
  // A node's "value" is the max of its incoming and outgoing incident flow (terminal-stage
  // nodes only have one side); this drives both the aggregation decision and the y-sort.
  const inFlow=new Array(rawNodes.length).fill(0);
  const outFlow=new Array(rawNodes.length).fill(0);
  for(const [kk,c] of linkMap){ const [s,t]=kk.split(">").map(Number); outFlow[s]+=c; inFlow[t]+=c; }
  const nodeFlow=rawNodes.map((n,i)=>Math.max(inFlow[i],outFlow[i]));

  // -- Decide which raw nodes collapse into their stage's Other node. --
  // Below-threshold nodes collapse; at/above-threshold survive. Threshold of 1 keeps all.
  const collapse=new Array(rawNodes.length).fill(false);
  const otherTaxa=new Map();   // stage -> count of distinct taxa folded into Other
  for(let i=0;i<rawNodes.length;i++){
    if(minFlow>1 && nodeFlow[i]<minFlow){
      collapse[i]=true;
      otherTaxa.set(rawNodes[i].stage,(otherTaxa.get(rawNodes[i].stage)||0)+1);
    }
  }

  // -- Build the final aggregated node list: surviving raw nodes + one Other per stage that needs it. --
  const nodes=[]; const remap=new Array(rawNodes.length).fill(-1);
  const otherNode=new Map();   // stage -> aggregated node index
  for(let i=0;i<rawNodes.length;i++){
    if(collapse[i]) continue;
    remap[i]=nodes.length; nodes.push({stage:rawNodes[i].stage,label:rawNodes[i].label,other:false});
  }
  for(const [stage,n] of otherTaxa){
    otherNode.set(stage,nodes.length);
    nodes.push({stage:stage,label:OTHER_PREFIX+n+" taxa)",other:true});
  }
  function finalId(rawId){ return collapse[rawId] ? otherNode.get(rawNodes[rawId].stage) : remap[rawId]; }

  // -- Reroute every link through finalId, summing values and merging lineage tallies. --
  const aggVal=new Map();       // "fs>ft" -> summed count
  const aggLineage=new Map();   // "fs>ft" -> merged Map(root_class -> count)
  for(const [kk,c] of linkMap){
    const [rs,rt]=kk.split(">").map(Number);
    const fs=finalId(rs), ft=finalId(rt);
    const fk=fs+">"+ft;
    aggVal.set(fk,(aggVal.get(fk)||0)+c);
    let dst=aggLineage.get(fk); if(!dst){dst=new Map();aggLineage.set(fk,dst);}
    mergeTally(dst,linkLineage.get(kk));
  }

  const source=[],target=[],value=[],linkClass=[];
  for(const [fk,c] of aggVal){
    const [s,t]=fk.split(">").map(Number);
    source.push(s);target.push(t);value.push(c);
    // link color = majority root_class of records on this edge
    linkClass.push(argmaxKey(aggLineage.get(fk)) || LINEAGE_UNKNOWN);
  }
  return {nodes,link:{source,target,value,linkClass},stages,used,nodeFlow:null};
}

function draw(){
  document.getElementById('minLbl').textContent=document.getElementById('minFlow').value;
  const b=build();
  const chart=document.getElementById('chart');
  if(b.empty){ Plotly.purge(chart); chart.innerHTML='<p style="color:#6b7280;padding:40px">Enable at least two stages.</p>'; return; }

  const nStage=b.stages.length;

  // Per-node total flow (max of incident in/out) for sizing/sorting the aggregated nodes.
  const inF=new Array(b.nodes.length).fill(0), outF=new Array(b.nodes.length).fill(0);
  for(let i=0;i<b.link.source.length;i++){ outF[b.link.source[i]]+=b.link.value[i]; inF[b.link.target[i]]+=b.link.value[i]; }
  const flow=b.nodes.map((n,i)=>Math.max(inF[i],outF[i]));

  // Group node indices by stage, then order each stage by value DESC (Other/blank pinned bottom).
  const byStage=new Map();
  b.nodes.forEach((n,i)=>{ if(!byStage.has(n.stage)) byStage.set(n.stage,[]); byStage.get(n.stage).push(i); });
  let maxNodesInStage=1;
  const ys=new Array(b.nodes.length).fill(0.5);
  for(const [stage,idxs] of byStage){
    maxNodesInStage=Math.max(maxNodesInStage,idxs.length);
    const pinned=i=> b.nodes[i].other || b.nodes[i].label===EMPTY;   // sink to bottom
    idxs.sort((a,c)=>{
      const pa=pinned(a)?1:0, pc=pinned(c)?1:0;
      if(pa!==pc) return pa-pc;            // non-pinned first
      return flow[c]-flow[a];              // heaviest at top
    });
    const tot=idxs.reduce((s,i)=>s+flow[i],0) || 1;
    let cum=0;
    for(const i of idxs){
      // y = cumulative fraction at the node's center, so heavy flows cluster near the top.
      ys[i]=Math.min(0.98,Math.max(0.02,(cum+flow[i]/2)/tot));
      cum+=flow[i];
    }
  }

  const nodeColors=b.nodes.map(n=> (n.other || n.label===EMPTY) ? MUTED : STAGE_COLORS[n.stage%STAGE_COLORS.length]);
  const nodeLabels=b.nodes.map(n=>n.label);
  // link color = majority root_class of records on this edge (carried as b.link.linkClass)
  const linkColors=b.link.linkClass.map(rc=>{
    const base=LINEAGE_COLORS.get(rc)||MUTED;
    return hexA(base.startsWith('#')?base:'#aab3bf',0.45);
  });

  const xs=b.nodes.map(n=> nStage>1 ? Math.min(0.96,Math.max(0.04,n.stage/(nStage-1))) : 0.5);

  // Scale chart height to the busiest visible stage so dense ranks get vertical room.
  const height=Math.max(600, maxNodesInStage*22);
  chart.style.height=height+"px";

  const trace={
    type:"sankey", arrangement:"fixed",
    node:{ pad:12, thickness:16, line:{color:"#ffffff",width:0.8},
           label:nodeLabels, color:nodeColors, x:xs, y:ys,
           hovertemplate:"%{label}<br>%{value} rows<extra></extra>" },
    link:{ source:b.link.source, target:b.link.target, value:b.link.value, color:linkColors,
           hovertemplate:"%{source.label} → %{target.label}<br>%{value} rows<extra></extra>" }
  };
  const ann=b.stages.map((k,i)=>({
    x: nStage>1? i/(nStage-1):0.5, y:1.04, xref:"paper", yref:"paper",
    text:"<b>"+LABELS[k]+"</b>", showarrow:false, font:{color:"#374151",size:11},
    xanchor: i===0?"left":(i===nStage-1?"right":"center")
  }));
  const layout={
    height:height,
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{color:"#1a1a1a",size:11}, margin:{l:10,r:10,t:34,b:10}, annotations:ann
  };
  Plotly.react(chart,[trace],layout,{responsive:true,displayModeBar:true,
    modeBarButtonsToRemove:['lasso2d','select2d']});
}

document.getElementById('hideEmpty').addEventListener('change',draw);
document.getElementById('minFlow').addEventListener('input',draw);
document.querySelectorAll('[data-preset]').forEach(btn=>btn.addEventListener('click',()=>{
  selected=btn.dataset.preset.split(',').filter(k=>AVAIL.has(k)); renderDims(); draw();
}));

renderLegend(); renderDims(); draw();
window.addEventListener('resize',()=>Plotly.Plots.resize(document.getElementById('chart')));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
