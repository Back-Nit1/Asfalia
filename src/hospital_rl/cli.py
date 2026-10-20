import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital-rl")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "train", "evaluate", "adapt", "benchmark", "counterfactual",
                 "replay", "orchestrate", "dashboard", "cloud-plan", "cleanup",
                 "export-policy", "benchmark-inference"):
        sub.add_parser(name)
    args, _ = p.parse_known_args(argv)
    print(f"hospital-rl {args.cmd}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
