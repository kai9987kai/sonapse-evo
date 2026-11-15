"""Standalone HTML inference runner.

``export_standalone_html(package)`` produces one self-contained HTML file (no network, no dependencies) that
embeds a frozen model and runs it locally in a browser.

What the runner executes (frozen inference)
    r <- clip((1 - leak) r + leak * max(0, tanh(W_eff r + B s + bias)), 0, 1)        for ``microsteps`` per step
with ``W_eff`` = base weights + learned weave deltas, lesion mask and pruning applied, grown synapses
included; the motor read-out when an agent is exported; and the frozen Micro-Collective next-sensory
prediction when one was trained.

What it does NOT execute, and says so on the page in a capability table: multi-compartment dynamics,
plasticity, structural change, neuromodulation, memory stores, live camera / microphone input and the
embodied environment.  A model that relied on those is reduced to its frozen scalar pathway; the runner
never silently pretends otherwise.  ``reference_inference`` is the same computation in Python, used to
verify the JavaScript against this code base.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .package import _arr, validate_model_package

RUNNER_FORMAT = "SONAPSE_STANDALONE_MODEL"
RUNNER_VERSION = "1.0.0"

#: The JavaScript runtime (also loadable under Node for parity tests: it only needs ``MODEL`` to be defined).
RUNTIME_JS = r"""
class SonapseRuntime {
  constructor(model) {
    this.m = model; this.n = model.neurons; this.steps = 0;
    this.r = Float64Array.from(model.r);
  }
  reset() { this.r = Float64Array.from(this.m.r); this.steps = 0; }
  microstep(obs, aux) {
    const m = this.m, n = this.n, drive = new Float64Array(n), e = m.edges, b = m.B;
    for (let k = 0; k < e.pre.length; k++) drive[e.post[k]] += this.r[e.pre[k]] * e.w[k];
    for (let k = 0; k < b.rows.length; k++) drive[b.rows[k]] += b.values[k] * (obs[b.cols[k]] || 0);
    const next = new Float64Array(n), cap = m.current_cap;
    for (let i = 0; i < n; i++) {
      let a = aux ? Math.max(-cap, Math.min(cap, aux[i] || 0)) : 0;
      const act = Math.max(0, Math.tanh(drive[i] + a + m.bias[i]));
      next[i] = Math.max(0, Math.min(1, (1 - m.leak) * this.r[i] + m.leak * act));
      if (m.node_mask) next[i] *= m.node_mask[i];
    }
    this.r = next;
  }
  step(obs, aux) {
    if (!Array.isArray(obs) || obs.length !== this.m.sensory_dim || obs.some(v => !Number.isFinite(v)))
      throw new Error('Provide exactly ' + this.m.sensory_dim + ' finite observation values.');
    for (let k = 0; k < this.m.microsteps; k++) this.microstep(obs, aux);
    this.steps++;
    return this.r;
  }
  action() {
    const a = this.m.agent; if (!a) return null;
    const out = [];
    for (let i = 0; i < a.weights.length; i++) {
      let z = 0; for (let j = 0; j < a.width; j++) z += a.weights[i][j] * this.r[a.offset + j];
      out.push(Math.tanh(z));
    }
    return out;
  }
  meanRate() { let s = 0; for (let i = 0; i < this.n; i++) s += this.r[i]; return s / this.n; }
  regionRates() {
    const sums = {}, counts = {};
    for (let i = 0; i < this.n; i++) { const g = this.m.regions[i]; sums[g] = (sums[g] || 0) + this.r[i]; counts[g] = (counts[g] || 0) + 1; }
    return Object.keys(sums).map(g => ({ region: this.m.region_names[g] || g, rate: sums[g] / counts[g] }));
  }
  predictSensory(x) {
    const mc = this.m.microCollective; if (!mc) return null;
    const logits = mc.router.map((row, i) => row.reduce((a, w, j) => a + w * x[j], mc.routerBias[i]));
    const mx = Math.max(...logits), probs = logits.map(z => Math.exp(z - mx)), den = probs.reduce((a, b) => a + b, 0);
    const gated = probs.map((p, i) => p / den * mc.gate[i]);
    const order = gated.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p || a.i - b.i).slice(0, mc.topK).filter(q => q.p > 0);
    const total = order.reduce((a, q) => a + q.p, 0) || 1, out = new Array(mc.dimension).fill(0);
    for (const q of order) for (let j = 0; j < mc.dimension; j++) {
      let z = mc.c[q.i][j]; for (let k = 0; k < mc.dimension; k++) z += mc.W[q.i][j][k] * x[k];
      out[j] += q.p / total * Math.tanh(z);
    }
    return out;
  }
}
if (typeof module !== 'undefined') module.exports = { SonapseRuntime };
"""

_NOT_EXECUTED = {
    "cell": "multi-compartment / multi-soma dynamics",
    "transmission": "delay lines, adaptive routing and uncertainty gating",
    "learning": "synaptic plasticity (weights are frozen at their exported values)",
    "structure": "structural plasticity and neurogenesis",
    "modulation": "neuromodulation and affective control",
    "memory": "memory stores, fast weights and replay",
    "prediction": "world model, prediction horizons and self-model",
    "sensory": "live sensory channels (camera / microphone are never requested)",
    "representation": "attention, FlyHash and NeuroWave packets",
    "connectome": "adaptive region routing",
    "execution": "tiled / quantised execution experiments",
}


def build_runner_model(package: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a validated model package to the compact frozen model the runner executes."""
    report = validate_model_package(package)
    if not report["ok"]:
        raise ValueError("cannot build a standalone runner from an invalid package: " + "; ".join(report["issues"]))
    topo, sub, weave = package["topology"], package["substrate"], package.get("weave")
    n = int(topo["total_neurons"])
    pre = _arr(sub["edges"]["pre"]).astype(np.int64)
    post = _arr(sub["edges"]["post"]).astype(np.int64)
    w = _arr(sub["edges"]["w"]).astype(np.float64)
    mask = _arr(sub["lesion_mask"]).astype(np.float64)
    node_mask = None
    plastic = sub.get("plastic") or {}
    if plastic and len(_arr(plastic.get("w", []))):
        pre = np.concatenate([pre, _arr(plastic["pre"]).astype(np.int64)])
        post = np.concatenate([post, _arr(plastic["post"]).astype(np.int64)])
        w = np.concatenate([w, _arr(plastic["w"]).astype(np.float64)])
    enabled: List[str] = []
    micro = None
    if weave:
        syn = weave["synapses"]
        grown_w = _arr(syn.get("grown_w", [])).astype(np.float64)
        if len(grown_w):
            pre = np.concatenate([pre, _arr(syn["grown_pre"]).astype(np.int64)])
            post = np.concatenate([post, _arr(syn["grown_post"]).astype(np.int64)])
            w = np.concatenate([w, grown_w])
        if weave.get("node_mask") is not None:
            node_mask = _arr(weave["node_mask"]).astype(np.float64)
        weave_active = any(bool(v) for v in weave["config"]["enabled"].values()) or node_mask is not None
        if weave_active:
            # exactly WeaveSubstrate.effective_weights: (anchor + learned delta) * extended lesion mask * alive
            delta, alive = _arr(syn["w_delta"]).astype(np.float64), _arr(syn["alive"]).astype(np.float64)
            extended = np.concatenate([mask, np.ones(len(w) - len(mask))])
            w = (w + delta) * extended * alive
            mask = None
        enabled = sorted(k for k, v in weave["config"]["enabled"].items() if v)
        mc_state = (weave.get("mechanisms") or {}).get("micro_collective")
        if mc_state and mc_state["snapshot"].get("trained"):
            snap = mc_state["snapshot"]
            micro = {"dimension": snap["dimension"], "topK": snap["topK"] if snap["expertRouting"] else 1, "W": snap["experts"]["W"],
                     "c": snap["experts"]["c"], "router": snap["router"], "routerBias": snap["routerBias"], "gate": snap["gate"],
                     "validationLoss": snap.get("validationLoss")}
    if mask is not None:                                     # legacy path: only base edges carry the lesion mask
        w = w * np.concatenate([mask, np.ones(len(w) - len(mask))])
    live = w != 0
    agent = None
    if package.get("agent"):
        weights = _arr(package["agent"]["motor_head"]).astype(np.float64)
        width = int(weights.shape[1])
        offset = int(sub.get("motor_offset", 512))
        agent = {"weights": weights.tolist(), "width": width, "offset": offset if offset + width <= n else 0}
    from ..registry import MECHANISMS

    groups = sorted({MECHANISMS[m].group for m in enabled if m in MECHANISMS})
    capabilities = {
        "frozen_inference": True, "motor_readout": agent is not None, "micro_collective_prediction": micro is not None,
        "continued_learning": False, "full_simulation": False, "live_sensory_operation": False, "exported_memory": False,
        "persistent_model_state": "starts from the exported rate state; changes are not saved",
        "not_executed": [_NOT_EXECUTED[g] for g in groups if g in _NOT_EXECUTED],
        "reduced_from_mechanisms": enabled,
    }
    b = sub["B"]
    return {"format": RUNNER_FORMAT, "version": RUNNER_VERSION, "sourceFormatVersion": package.get("formatVersion"),
            "sourceDigest": (package.get("integrity") or {}).get("digest"), "neurons": n,
            "sensory_dim": int(sub["sensory_dim"]), "leak": float(sub["leak"]), "microsteps": int(sub["microsteps"]),
            "current_cap": float(sub["current_cap"]), "r": _arr(sub["r"]).astype(np.float64).tolist(),
            "bias": _arr(sub["bias"]).astype(np.float64).tolist(),
            "B": {"rows": _arr(b["rows"]).astype(int).tolist(), "cols": _arr(b["cols"]).astype(int).tolist(),
                  "values": _arr(b["values"]).astype(np.float64).tolist()},
            "edges": {"pre": pre[live].tolist(), "post": post[live].tolist(), "w": w[live].tolist()},
            "node_mask": None if node_mask is None else node_mask.tolist(),
            "regions": _arr(topo["region_assignments"]).astype(int).tolist(), "region_names": list(topo["region_names"]),
            "topology_kind": topo.get("kind", "legacy"), "agent": agent, "microCollective": micro, "capabilities": capabilities}


def reference_inference(model: Dict[str, Any], observations: Sequence[Sequence[float]]) -> np.ndarray:
    """Python reference of exactly what the standalone runner computes.  Returns rates of shape (T, N)."""
    n = model["neurons"]
    r = np.asarray(model["r"], dtype=np.float64)
    pre, post, w = (np.asarray(model["edges"][k]) for k in ("pre", "post", "w"))
    rows, cols, vals = (np.asarray(model["B"][k]) for k in ("rows", "cols", "values"))
    bias, leak = np.asarray(model["bias"], dtype=np.float64), float(model["leak"])
    out = []
    for obs in observations:
        obs = np.asarray(obs, dtype=np.float64)
        for _ in range(int(model["microsteps"])):
            drive = np.zeros(n)
            np.add.at(drive, post, r[pre] * w)
            np.add.at(drive, rows, vals * obs[cols])
            r = np.clip((1.0 - leak) * r + leak * np.maximum(0.0, np.tanh(drive + bias)), 0.0, 1.0)
            if model.get("node_mask") is not None:
                r = r * np.asarray(model["node_mask"], dtype=np.float64)
        out.append(r.copy())
    return np.stack(out) if out else np.zeros((0, n))


def _script_safe(text: str) -> str:
    return (text.replace("</", "<\\/").replace("<!--", "<\\!--").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def export_standalone_html(package: Dict[str, Any], path: Optional[str] = None, title: str = "SONAPSE-EVO standalone model") -> str:
    """Return (and optionally write) a self-contained HTML inference page for ``package``."""
    model = build_runner_model(package)
    caps = model["capabilities"]
    esc = html.escape
    rows = [("Frozen recurrent inference", "yes"), ("Motor read-out", "yes" if caps["motor_readout"] else "not exported"),
            ("Micro-Collective next-sensory prediction", "yes (frozen)" if caps["micro_collective_prediction"] else "not trained / not exported"),
            ("Continued learning", "no - import the JSON package into the Python runtime"),
            ("Full simulation with environment", "no - Python runtime only"),
            ("Live camera / microphone", "no - never requested; values are typed in manually"),
            ("Exported memory stores", "no - kept in the JSON package only"),
            ("Persistent state", esc(caps["persistent_model_state"]))]
    rows += [("Not executed here", esc(item)) for item in caps["not_executed"]]
    table = "".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in rows)
    default_obs = ", ".join(["0.3", "0.5", "0.8"] + ["0"] * (model["sensory_dim"] - 3)) if model["sensory_dim"] >= 3 else "0"
    embedded = _script_safe(json.dumps(model, separators=(",", ":")))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>
:root{{--bg:#0a101c;--panel:#121e2e;--line:#2d415c;--text:#e8f0fb;--muted:#95a8be;--accent:#64e1dc}}
@media (prefers-color-scheme: light){{:root{{--bg:#f4f7fb;--panel:#ffffff;--line:#c9d5e4;--text:#132033;--muted:#55677d;--accent:#0b7f86}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif;padding:16px}}
.wrap{{max-width:920px;margin:auto}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}}
textarea{{width:100%;min-height:84px;box-sizing:border-box;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px;font:12px ui-monospace,monospace}}
button{{background:var(--accent);color:var(--bg);border:0;padding:9px 14px;border-radius:8px;cursor:pointer;margin:6px 6px 0 0;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:12px}}td{{border-bottom:1px solid var(--line);padding:5px 4px;vertical-align:top}}
.muted{{color:var(--muted);font-size:12px}}.metric{{font-size:26px;color:var(--accent);font-weight:700}}pre{{white-space:pre-wrap;font-size:12px;margin:8px 0 0}}
</style></head><body><div class="wrap">
<h1>{esc(title)}</h1>
<p class="muted">Frozen model snapshot &middot; {model["neurons"]} neurons &middot; {len(model["edges"]["w"])} synapses &middot; {esc(str(model["topology_kind"]))} topology &middot; runs locally, no network access</p>
<div class="card"><h2>Inference</h2><div class="metric" id="headline">&mdash;</div><p class="muted" id="status">Enter {model["sensory_dim"]} comma-separated observation values and step the model.</p>
<label for="obs" class="muted">Observation vector</label><textarea id="obs">{default_obs}</textarea>
<button id="step">Run one step</button><button id="run10">Run 10 steps</button><button id="reset">Reset state</button>
<pre id="detail"></pre></div>
<div class="card" id="mcCard" hidden><h2>Micro-Collective prediction (frozen)</h2><p class="muted">Predicts the next sixteen-channel sensory vector from the first sixteen observation values.</p><button id="mc">Predict next sensory state</button><pre id="mcOut"></pre></div>
<div class="card"><h2>What this runner does and does not do</h2><table>{table}</table>
<p class="muted">Source package digest: {esc(str(model["sourceDigest"]))}. A digest detects modification; it is not a signature. This page is a research artefact: a synthetic recurrent model, not a biological simulation and not a calibrated predictor.</p></div>
</div>
<script>const MODEL={embedded};
{RUNTIME_JS}
const rt=new SonapseRuntime(MODEL),$=id=>document.getElementById(id);
function parse(){{const v=$('obs').value.split(',').map(s=>Number(s.trim()));if(v.length!==MODEL.sensory_dim||v.some(x=>!Number.isFinite(x)))throw new Error('Provide exactly '+MODEL.sensory_dim+' finite numbers.');return v;}}
function show(){{$('headline').textContent='mean rate '+rt.meanRate().toFixed(5);const a=rt.action();let t='steps: '+rt.steps+'\\n'+rt.regionRates().map(q=>q.region+': '+q.rate.toFixed(5)).join('\\n');if(a)t+='\\naction: ['+a.map(x=>x.toFixed(4)).join(', ')+']';$('detail').textContent=t;}}
function run(k){{try{{const o=parse();for(let i=0;i<k;i++)rt.step(o);$('status').textContent='Inference complete.';show();}}catch(e){{$('status').textContent=e.message;}}}}
$('step').addEventListener('click',()=>run(1));$('run10').addEventListener('click',()=>run(10));
$('reset').addEventListener('click',()=>{{rt.reset();$('status').textContent='State reset to the exported snapshot.';show();}});
if(MODEL.microCollective){{$('mcCard').hidden=false;$('mc').addEventListener('click',()=>{{try{{const x=parse().slice(0,MODEL.microCollective.dimension);$('mcOut').textContent=rt.predictSensory(x).map((v,i)=>'ch'+String(i+1).padStart(2,'0')+': '+v.toFixed(4)).join('\\n');}}catch(e){{$('mcOut').textContent=e.message;}}}});}}
show();
</script></body></html>"""
    if path is not None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(page, encoding="utf-8")
    return page


__all__ = ["RUNNER_FORMAT", "RUNNER_VERSION", "RUNTIME_JS", "build_runner_model", "reference_inference", "export_standalone_html"]
