"""Live training dashboard: reads runs/*/metrics.jsonl written by train_lora.py, plus nvidia-smi.

    python scripts/dashboard.py [--port 8080]
    ssh -L 8080:localhost:8080 <gpu-box>      # then open http://localhost:8080

Standard library only, no CDN, so it works on a box with no outbound access. Binds to localhost by default.
"""

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from bhojpuri_tts import REPO_ROOT

RUNS = REPO_ROOT / "runs"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bhojpuri LoRA training</title>
<style>
:root{color-scheme:light;--bg:#fafaf9;--card:#fff;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--a:#2563eb;--b:#dc2626;--c:#16a34a;--d:#9333ea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0c0a09;--card:#1c1917;--fg:#fafaf9;--mut:#a8a29e;--line:#292524;--a:#60a5fa;--b:#f87171;--c:#4ade80;--d:#c084fc}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0c0a09;--card:#1c1917;--fg:#fafaf9;--mut:#a8a29e;--line:#292524;--a:#60a5fa;--b:#f87171;--c:#4ade80;--d:#c084fc}
[hidden]{display:none!important}.note{margin:-6px 0 12px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0}header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.badge{padding:2px 10px;border-radius:99px;font-weight:600;font-size:12px;border:1px solid currentColor}
.running{color:var(--c)}.done{color:var(--a)}.stalled{color:var(--b)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.k{color:var(--mut);font-size:12px}.v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}.s{color:var(--mut);font-size:12px}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden;margin:8px 0}.bar>i{display:block;height:100%;background:var(--a)}
.charts{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))}
canvas{width:100%;height:190px;display:block}h2{font-size:13px;margin:0 0 6px}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:3px 8px;border-bottom:1px solid var(--line)}
.wrap{overflow-x:auto}audio{height:28px;width:220px}
.planned{color:var(--mut)}.you{color:var(--d)}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
</style></head><body>
<header><h1>Bhojpuri LoRA training <span id="status" class="badge"></span></h1>
<label>Run <select id="run"></select></label></header>
<div id="snapnote" class="s note" hidden></div>
<div class="card"><div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px"><b id="prog"></b><span id="eta" class="s"></span></div>
<div class="bar"><i id="barfill" style="width:0"></i></div></div>
<div class="grid" id="tiles" style="margin-top:12px"></div>
<div class="card" id="plan" style="margin-top:12px"><h2>Estimated timeline — whole plan</h2><div class="wrap" id="plantable"></div>
<div class="s" id="plannote" style="margin-top:6px"></div></div>
<div class="charts" id="charts" style="margin-top:12px"></div>
<div class="card" id="samplecard" style="margin-top:12px"><h2>Samples (listen by step — judge tone and prosody by ear)</h2><div class="wrap" id="samples"></div></div>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt=s=>{if(!isFinite(s))return"–";s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${m}m`:m?`${m}m ${s%60}s`:`${s}s`};
const med=a=>{if(!a.length)return NaN;a=[...a].sort((x,y)=>x-y);return a[a.length>>1]};
const CH=[["Training loss","loss","--a","train"],["Validation loss (EMA)","val_loss","--b","val"],["Seconds per update","sec_per_update","--d","train"],["Gradient norm","grad_norm","--c","train"],["Learning rate","lr","--a","train"],["Audio-hours per hour (speed)","speed","--c","train"]];
function chart(cv,pts,color){
  const dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
  const g=cv.getContext("2d");g.scale(dpr,dpr);g.clearRect(0,0,W,H);
  g.font="11px system-ui";g.fillStyle=css("--mut");g.strokeStyle=css("--line");
  if(pts.length<1){g.fillText("waiting for data…",10,20);return}
  const L=46,R=8,T=8,B=20;let xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(x1==x0)x1=x0+1;if(y1==y0){y1+=1;y0-=1}const pad=(y1-y0)*.05;y0-=pad;y1+=pad;
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  for(let i=0;i<=4;i++){const y=y0+(y1-y0)*i/4;g.beginPath();g.moveTo(L,Y(y));g.lineTo(W-R,Y(y));g.stroke();
    g.fillText(Math.abs(y)<.01&&y!=0?y.toExponential(1):(+y.toPrecision(3)).toString(),2,Y(y)+4)}
  g.fillText(x0,L,H-5);g.textAlign="right";g.fillText("update "+x1,W-R,H-5);g.textAlign="left";
  g.strokeStyle=css(color);g.lineWidth=1.8;g.beginPath();pts.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();
  if(pts.length<40){g.fillStyle=css(color);pts.forEach(p=>{g.beginPath();g.arc(X(p[0]),Y(p[1]),2.5,0,7);g.fill()})}
}
function tile(k,v,s){return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s||""}</div></div>`}
// Whole-plan estimate. 0.44 s/update was measured on the A100 with real batches (probe, 27.9 GB peak); a running
// run's own median replaces it. Only the 10h config's 8,000 updates is decided; the other update counts are assumptions.
const SPU0=0.44;
const PLAN=[
 {n:"Stock baseline on 32 diagnostic clips",done:1,t:120},
 {n:"Probe, 200-step overfit check, uploader test",done:1,t:180},
 {n:"Slice run (1.9h, pipeline check)",run:"lora_slice_r32",up:3000},
 {n:"Diagnostics on the slice checkpoint",t:120},
 {n:"10h run: baseline config",run:"lora_10h_r32",up:8000},
 {n:"10h sweep: text embedding unfrozen",up:8000},
 {n:"10h sweep: rank 64",up:8000},
 {n:"10h sweep: rank 16",up:8000},
 {n:"10h sweep: learning-rate variants (2 runs)",up:16000,assume:1},
 {n:"Diagnostics for the 6 sweep checkpoints",t:720},
 {n:"Stop/go gate: diagnostics + native-speaker listening",manual:1},
 {n:"Full 90.7h run (only if the gate passes)",up:30000,assume:1}];
function plan(cur,tr,last,spuNow){
  const spu=isFinite(spuNow)?spuNow:SPU0;let toGate=0,afterGate=0,gated=false,rows="";
  PLAN.forEach(p=>{let secs=0,state="planned",label="planned",est;
    if(p.done){state=label="done";est="took "+fmt(p.t)}
    else if(p.manual){state="you";label="your call";est="human step";gated=true}
    else if(p.t){secs=p.t;est="~"+fmt(secs)}
    else if(p.run&&p.run===cur&&last){const fin=last.kind=="done",u=tr.length?tr[tr.length-1].update:0;
      secs=fin?0:Math.max(p.up-u,0)*spu;state=label=fin?"done":"running";est=fin?"done":"~"+fmt(secs)+" left"}
    else{secs=p.up*spu;est="~"+fmt(secs)+(p.assume?" (assumed)":"")}
    if(!p.manual){if(gated)afterGate+=secs;else toGate+=secs}
    rows+=`<tr><td>${p.n}</td><td class="n">${p.up?p.up.toLocaleString()+" updates":""}</td><td class="n">${est}</td><td><span class="badge ${state}">${label}</span></td></tr>`});
  $("plantable").innerHTML="<table>"+rows+"</table>";
  $("plannote").textContent=`Remaining to the stop/go gate ≈ ${fmt(toGate)}; the full run adds ≈ ${fmt(afterGate)} if the gate passes. Based on ${spu.toFixed(2)} s/update ${isFinite(spuNow)?"(this run)":"(measured on the A100)"}. Excludes model loading and time spent waiting on you; update counts other than the 10h config's are assumptions.`}
let gpu={},recs=[],cur="";
async function loadRuns(){
  const runs=await (await fetch("/api/runs")).json();const sel=$("run");
  if(JSON.stringify(runs)!=sel.dataset.r){sel.dataset.r=JSON.stringify(runs);const keep=sel.value;
    sel.innerHTML="";runs.forEach(r=>{const o=document.createElement("option");o.value=o.textContent=r;sel.append(o)});
    sel.value=runs.includes(keep)?keep:runs[0]||""}
  cur=sel.value;
}
async function tick(){
  try{await loadRuns();if(!cur){$("prog").textContent="No runs with metrics.jsonl yet — start train_lora.py";plan("",[],null,NaN);return}
  const [d,gp]=await Promise.all([fetch("/api/run?name="+encodeURIComponent(cur)).then(r=>r.json()),fetch("/api/gpu").then(r=>r.json())]);
  recs=d;gpu=gp;render()}catch(e){$("status").textContent="dashboard offline";$("status").className="badge stalled"}
}
function render(){
  const tr=recs.filter(r=>r.kind=="train"),va=recs.filter(r=>r.kind=="val"),ck=recs.filter(r=>r.kind=="checkpoint");
  const st=[...recs].reverse().find(r=>r.kind=="start"),last=recs[recs.length-1],lt=tr[tr.length-1];
  if(!last){plan(cur,[],null,NaN);return}
  const max=(st&&st.max_updates)||(lt&&lt.max_updates)||0,upd=lt?lt.update:(st?st.update:0);
  const recent=tr.slice(-10).map(r=>r.sec_per_update),spu=med(recent);plan(cur,tr,last,spu);
  const rem=Math.max(max-upd,0),eta=rem*spu,age=Date.now()/1000-last.t,done=last.kind=="done"||(max&&upd>=max);
  const stalled=!done&&age>Math.max(180,(lt?lt.sec_per_update*60:0));
  $("status").textContent=done?"finished":stalled?"stopped / stalled":"running";$("status").className="badge "+(done?"done":stalled?"stalled":"running");
  $("prog").textContent=`update ${upd.toLocaleString()} / ${max.toLocaleString()}  (${max?(100*upd/max).toFixed(1):0}%)`;
  $("barfill").style.width=(max?100*upd/max:0)+"%";
  const finish=new Date(Date.now()+eta*1000);
  $("eta").textContent=done?"complete":isFinite(eta)?`ETA ${fmt(eta)} · finishes ~${finish.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}`:"ETA: waiting for data";
  // active training time = sum of logged windows (survives resume gaps)
  let active=0,prev=st?st.update:0;tr.forEach(r=>{if(r.update>prev){active+=r.sec_per_update*(r.update-prev)}prev=r.update});
  const wall=(st?Date.now()/1000-st.t:0),lastVal=va[va.length-1],bestVal=va.length?va.reduce((a,b)=>b.val_loss<a.val_loss?b:a):null;
  const g=gpu.gpus&&gpu.gpus[0];
  $("tiles").innerHTML=[
    tile("Time this session",fmt(done?active:wall),"active training "+fmt(active)),
    tile("Time remaining",done?"0s":fmt(eta),isFinite(spu)?`${spu.toFixed(2)} s/update (median of last 10 logs)`:""),
    tile("Loss",lt?lt.loss.toFixed(4):"–",lastVal?`val ${lastVal.val_loss.toFixed(4)} @ ${lastVal.update}`+(bestVal?` · best ${bestVal.val_loss.toFixed(4)} @ ${bestVal.update}`:""):"no validation yet"),
    tile("Epoch",lt?lt.epoch:"–",lt?`speed ${lt.speed.toFixed(1)}× realtime`:""),
    tile("Peak VRAM (torch)",lt?lt.vram_gb.toFixed(1)+" GB":"–",g?`GPU now ${g.mem_used_gb.toFixed(1)}/${g.mem_total_gb.toFixed(0)} GB`:""),
    tile("GPU",g?g.util+"%":"n/a",g?`${g.temp}°C · ${g.power.toFixed(0)} W · ${g.name}`:"nvidia-smi unavailable"),
    tile("Checkpoints",ck.length,ck.length?"last @ update "+ck[ck.length-1].update:"none yet")].join("");
  if(!$("charts").children.length)$("charts").innerHTML=CH.map((c,i)=>`<div class="card"><h2>${c[0]}</h2><canvas id="c${i}"></canvas></div>`).join("");
  CH.forEach((c,i)=>chart($("c"+i),(c[3]=="val"?va:tr).filter(r=>r[c[1]]!=null).map(r=>[r.update,r[c[1]]]),c[2]));
}
async function samples(){
  if(!cur)return;const d=await (await fetch("/api/samples?name="+encodeURIComponent(cur))).json();const el=$("samples");
  const key=JSON.stringify(d);if(el.dataset.k==key)return;el.dataset.k=key;
  if(!d.length){el.textContent="No samples yet — first ones appear at the first sample_every step.";return}
  el.innerHTML="<table>"+d.slice().reverse().map(s=>`<tr><th>step ${+s.step.replace("step_","")}</th>`+s.files.map(f=>`<td>${f.split("/").pop()}<br><audio controls preload="none" src="/audio/${encodeURI(cur+"/samples/"+s.step+"/"+f)}"></audio></td>`).join("")+"</tr>").join("")+"</table>";
}
$("run").onchange=()=>{cur=$("run").value;$("charts").innerHTML="";tick();samples()};
addEventListener("resize",()=>recs.length&&render());
tick();samples();setInterval(tick,5000);setInterval(samples,30000);
</script></body></html>"""


def read_metrics(name: str) -> list[dict]:
    path = (RUNS / name / "metrics.jsonl").resolve()
    if RUNS.resolve() not in path.parents or not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # trainer may be mid-write on the last line
            pass
    return rows


def gpu_status() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {"gpus": []}
    gpus = []
    for line in out.strip().splitlines():
        name, util, used, total, temp, power = [x.strip() for x in line.split(",")]
        gpus.append(dict(name=name, util=int(util), mem_used_gb=int(used) / 1024, mem_total_gb=int(total) / 1024,
                         temp=int(temp), power=float(power)))
    return {"gpus": gpus}


def list_samples(name: str) -> list[dict]:
    root = (RUNS / name / "samples").resolve()
    if RUNS.resolve() not in root.parents or not root.is_dir():
        return []
    return [dict(step=d.name, files=sorted(p.name for p in d.glob("*.wav"))) for d in sorted(root.glob("step_*"))]


# Snapshot mode: the page above, with its /api calls answered from embedded data and the clock frozen at
# snapshot time, so an hours-old snapshot shows the state it captured instead of "stalled". Publish the
# output as a claude.ai artifact; regenerate and republish to refresh.
SNAPSHOT_JS = r"""
Date.now=()=>SNAP.at*1000;
window.fetch=async u=>{const p=new URL(u,"http://x").pathname,j=v=>({json:async()=>v});
  return p==="/api/runs"?j(SNAP.run?[SNAP.run]:[]):p==="/api/run"?j(SNAP.records):p==="/api/gpu"?j(SNAP.gpu):j([])};
document.getElementById("snapnote").hidden=false;
document.getElementById("snapnote").textContent="Snapshot taken "+new Date(SNAP.at*1000).toLocaleString()+" — republished periodically, not live. Audio samples are only in the live dashboard."+(SNAP.note?" "+SNAP.note:"");
document.getElementById("samplecard").hidden=true;
"""


def build_snapshot(run: str, out: Path, note: str = ""):
    import re
    import time

    data = dict(at=time.time(), run=run, records=read_metrics(run) if run else [], gpu=gpu_status(), note=note)
    style = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    body = re.search(r"<body>(.*)</body>", PAGE, re.S).group(1)
    shim = "<script>window.SNAP=" + json.dumps(data).replace("</", "<\\/") + ";" + SNAPSHOT_JS + "</script>"
    out.write_text(f"<title>Bhojpuri Training Monitor</title>\n<style>{style}</style>\n{body.replace('<script>', shim + '<script>', 1)}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        name = q.get("name", [""])[0]
        if url.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/api/runs":
            self._json(sorted((p.parent.name for p in RUNS.glob("*/metrics.jsonl")),
                              key=lambda n: (RUNS / n / "metrics.jsonl").stat().st_mtime, reverse=True))
        elif url.path == "/api/run":
            self._json(read_metrics(name))
        elif url.path == "/api/gpu":
            self._json(gpu_status())
        elif url.path == "/api/samples":
            self._json(list_samples(name))
        elif url.path.startswith("/audio/"):
            path = (RUNS / unquote(url.path[len("/audio/"):])).resolve()
            if RUNS.resolve() in path.parents and path.suffix == ".wav" and path.is_file():
                self._send(path.read_bytes(), "audio/wav")
            else:
                self._send(b"not found", "text/plain", 404)
        else:
            self._send(b"not found", "text/plain", 404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only if the port is firewalled")
    parser.add_argument("--snapshot", type=Path, metavar="OUT.html",
                        help="write a self-contained snapshot page for --run (default: newest run) and exit")
    parser.add_argument("--run", help="run name for --snapshot")
    parser.add_argument("--note", default="", help="extra line for --snapshot, e.g. Claude's last monitor check")
    args = parser.parse_args()
    if args.snapshot:
        runs = sorted((p.parent.name for p in RUNS.glob("*/metrics.jsonl")),
                      key=lambda n: (RUNS / n / "metrics.jsonl").stat().st_mtime, reverse=True)
        run = args.run or (runs[0] if runs else "")  # no run yet is fine: the page then shows the plan only
        build_snapshot(run, args.snapshot, args.note)
        print(f"wrote {args.snapshot} for run {run or '(none yet)'}")
        raise SystemExit(0)
    print(f"Dashboard on http://{args.host}:{args.port}  (runs dir: {RUNS})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
