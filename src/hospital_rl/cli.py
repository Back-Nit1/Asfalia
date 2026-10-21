import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital-rl")
    sub = p.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--config", required=True)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--out", default="runs/demo/episode.json")
    rp = sub.add_parser("replay")
    rp.add_argument("--episode", required=True)
    cf = sub.add_parser("counterfactual")
    cf.add_argument("--episode", required=True)
    cf.add_argument("--mode", choices=["fixed-intent", "adaptive"], default="fixed-intent")
    cf.add_argument("--seeds", type=int, default=10)
    bm = sub.add_parser("benchmark")
    bm.add_argument("--config")
    bm.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    bm.add_argument("--episodes", type=int, default=20)
    bm.add_argument("--size", type=int, default=8)
    bm.add_argument("--out")
    bm.add_argument("--tie-sensitivity", action="store_true")
    for name in ("train", "evaluate", "adapt", "orchestrate", "dashboard", "cloud-plan", "cleanup",
                 "export-policy", "benchmark-inference"):
        sub.add_parser(name)
    args, rest = p.parse_known_args(argv)
    if args.cmd in _DELEGATE:
        return _delegate(args.cmd, rest)
    if args.cmd == "demo":
        return _demo(args)
    if args.cmd == "replay":
        return _replay(args)
    if args.cmd == "counterfactual":
        return _cf(args)
    if args.cmd == "benchmark":
        return _bench(args)
    print(f"hospital-rl {args.cmd}: not implemented yet", file=sys.stderr)
    return 2


_DELEGATE = {"train": "hospital_rl.training.schools", "dashboard": "hospital_rl.api.__main__",
             "orchestrate": "hospital_rl.orchestration.__main__", "cloud-plan": "hospital_rl.infra.plan",
             "evaluate": "hospital_rl.evaluation.transfer", "adapt": "hospital_rl.training.adapt",
             "export-policy": "hospital_rl.infra.inference_cli"}


def _delegate(cmd: str, rest: list[str]) -> int:
    import importlib

    return importlib.import_module(_DELEGATE[cmd]).main(rest) or 0


def _demo(args) -> int:
    import json
    import pathlib

    from .policies.scripted import GreedyDefender, ScriptedAttacker
    from .runner import run_episode
    from .schemas import load_scenario

    ep = run_episode(load_scenario(args.config), args.seed, ScriptedAttacker(), GreedyDefender())
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ep))
    print(json.dumps(ep["metrics"], indent=1), f"\nwrote {out}")
    return 0


def _replay(args) -> int:
    import json

    ep = json.load(open(args.episode))
    print(f"{ep['scenario']} seed={ep['seed']} tie={ep['tie_policy']} return={ep['return']}")
    for st in ep["steps"]:
        a, d = st["attacker"], st["defender"]
        ev = [e for e in st["events"] if e["visible"] != "evaluator" and e["type"] != "attacker_outcome"]
        print(f"t={st['tick']:>3} A:{a['type']}({a['node']}) D:{d['type']}({d['node']}) "
              f"rD={st['reward']['defender']:+.3f} events={len(ev)}")
    print(json.dumps(ep["metrics"]))
    return 0


def _cf(args) -> int:
    import json

    from .evaluation import counterfactual as cf
    from .policies.scripted import GreedyDefender, Noop, ScriptedAttacker, StaticHardening
    from .schemas import ScenarioSpec

    ep = json.load(open(args.episode))
    defs = {"noop": Noop, "static_hardening": StaticHardening, "greedy": GreedyDefender}
    if args.mode == "fixed-intent":
        out = cf.fixed_intent(ep, defs)
    else:
        out = cf.adaptive(ScenarioSpec.model_validate(ep["spec"]), list(range(args.seeds)), ScriptedAttacker, defs)
    print(json.dumps(out, indent=1))
    return 0


def _bench(args) -> int:
    import json

    from .evaluation import benchmark

    if args.tie_sensitivity:
        print(json.dumps(benchmark.tie_sensitivity(args.split, args.episodes, args.size), indent=1))
        return 0
    r = benchmark.run(args.split, args.episodes, args.size)
    print(f"split={r['split']} hash={r['split_hash']} n={r['n_scenarios']}  (scripted attacker; uncalibrated)")
    print(f"{'defender':20}{'exposure':>10}{'crit_ok':>9}{'spend':>8}  feasible")
    for n, d in r["defenders"].items():
        print(f"{n:20}{d['weighted_exposure_frac']['mean']:>10.3f}{d['critical_success']['mean']:>9.3f}"
              f"{d['spend']['mean']:>8.1f}  {'yes' if d['feasible'] else 'NO (availability)'}")
    if args.out:
        open(args.out, "w").write(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
