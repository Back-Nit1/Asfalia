import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital-rl")
    sub = p.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--config", required=True)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--out", default="runs/demo/episode.json")
    for name in ("train", "evaluate", "adapt", "benchmark", "counterfactual",
                 "replay", "orchestrate", "dashboard", "cloud-plan", "cleanup",
                 "export-policy", "benchmark-inference"):
        sub.add_parser(name)
    args, _ = p.parse_known_args(argv)
    if args.cmd == "demo":
        return _demo(args)
    print(f"hospital-rl {args.cmd}: not implemented yet", file=sys.stderr)
    return 2


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


if __name__ == "__main__":
    raise SystemExit(main())
