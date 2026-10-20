"""Generate README figures from real simulator output (no plotting dependency). Run: uv run python scripts/make_figures.py"""
import json
import pathlib

from hospital_rl.evaluation import benchmark
from hospital_rl.runner import run_episode
from hospital_rl.policies.scripted import GreedyDefender, ScriptedAttacker
from hospital_rl.schemas import load_scenario

OUT = pathlib.Path("docs/img")
BG, FG, MUT, ACC, WARN = "#0f1419", "#e6edf3", "#8b98a5", "#4fb3d9", "#e5534b"
COL = {"ehr": "#4fb3d9", "pacs": "#6ec1a3", "workstation": "#b8a1e3", "billing": "#e3c36e", "idp": "#e39a6e",
       "vpn": "#e5534b", "k8s_service": "#8b98a5", "device": "#c78fb0"}
HEAD = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {{w}} {{h}}" font-family="ui-monospace,Menlo,monospace" font-size="12">' \
       f'<rect width="100%" height="100%" fill="{BG}"/>'


def topology():
    spec = load_scenario("configs/tiny.yaml")
    ep = run_episode(spec, 7, ScriptedAttacker(), GreedyDefender())
    zones = ["dmz", "clinical", "core", "business", "device"]
    pos, per = {}, {z: 0 for z in zones}
    for n in spec.nodes:
        pos[n.id] = (110 + zones.index(n.zone) * 170, 90 + per[n.zone] * 70)
        per[n.zone] += 1
    s = [HEAD.format(w=960, h=420)]
    for i, z in enumerate(zones):
        s.append(f'<rect x="{30 + i * 170}" y="40" width="160" height="340" rx="6" fill="none" stroke="#2a3441"/>'
                 f'<text x="{40 + i * 170}" y="58" fill="{MUT}">zone: {z}</text>')
    seen = set()
    for e in spec.edges:
        if e.kind != "reach" or (e.dst, e.src) in seen:
            continue
        seen.add((e.src, e.dst))
        (x1, y1), (x2, y2) = pos[e.src], pos[e.dst]
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#3a4757"/>')
    for n in spec.nodes:
        x, y = pos[n.id]
        s.append(f'<circle cx="{x}" cy="{y}" r="26" fill="{COL[n.asset_type]}"/>'
                 f'<text x="{x}" y="{y + 4}" text-anchor="middle" fill="#0f1419" font-size="11" font-weight="700">{n.id}</text>'
                 f'<text x="{x}" y="{y + 42}" text-anchor="middle" fill="{MUT}" font-size="10">{n.asset_type}'
                 f'{" · "+str(n.records)+" rec" if n.records else ""}{" · no patch" if not n.patchable else ""}</text>')
    m = ep["metrics"]
    s.append(f'<text x="30" y="404" fill="{FG}">tiny fixture, seed 7, scripted attacker vs greedy defender: exfiltrated '
             f'{m["records_exfiltrated"]} rec, critical success {m["critical_success"]:.2f}, spend {m["spend"]:.1f}</text></svg>')
    (OUT / "topology_tiny.svg").write_text("".join(s))


def tradeoff():
    r = benchmark.run("validation", 20, size=12)
    W, H, L, B = 640, 420, 70, 60
    s = [HEAD.format(w=W, h=H)]
    s.append(f'<text x="{L}" y="28" fill="{FG}">Security / availability tradeoff — validation, 20 scenarios, 12 nodes</text>')
    x = lambda v: L + v / 0.6 * (W - L - 30)                       # exposure 0..0.6
    y = lambda v: H - B - (v - 0.0) * (H - B - 50)                 # critical success 0..1
    s.append(f'<rect x="{L}" y="{y(1.0)}" width="{W - L - 30}" height="{y(0.95) - y(1.0)}" fill="#1c2b22"/>'
             f'<line x1="{L}" y1="{y(0.95)}" x2="{W - 30}" y2="{y(0.95)}" stroke="#6ec1a3" stroke-dasharray="4"/>'
             f'<text x="{W - 150}" y="{y(0.95) + 16}" fill="#6ec1a3">feasible ≥ 0.95</text>')
    s.append(f'<line x1="{L}" y1="{H - B}" x2="{W - 30}" y2="{H - B}" stroke="{MUT}"/><line x1="{L}" y1="{H - B}" x2="{L}" y2="50" stroke="{MUT}"/>'
             f'<text x="{L + 200}" y="{H - 20}" fill="{MUT}">sensitivity-weighted exposure (lower is better)</text>'
             f'<text x="14" y="{H // 2}" fill="{MUT}" transform="rotate(-90 14 {H // 2})">critical workflow success</text>')
    for t in (0, 0.2, 0.4, 0.6):
        s.append(f'<text x="{x(t)}" y="{H - B + 16}" fill="{MUT}" text-anchor="middle">{t}</text>')
    for t in (0, 0.5, 1.0):
        s.append(f'<text x="{L - 8}" y="{y(t) + 4}" fill="{MUT}" text-anchor="end">{t}</text>')
    for name, d in r["defenders"].items():
        cx, cy = x(d["weighted_exposure_frac"]["mean"]), y(d["critical_success"]["mean"])
        c = ACC if d["feasible"] else WARN
        s.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{c}"/><text x="{cx + 12}" y="{cy + 4}" fill="{FG}">{name}'
                 f'{"" if d["feasible"] else " (infeasible)"}</text>')
    s.append("</svg>")
    (OUT / "tradeoff_validation.svg").write_text("".join(s))
    (OUT / "benchmark_validation.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    topology()
    tradeoff()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
