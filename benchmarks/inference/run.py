"""Thin wrapper; the benchmark lives in hospital_rl.infra.inference_bench (`hospital-rl benchmark-inference`)."""
from hospital_rl.infra.inference_bench import main

if __name__ == "__main__":
    raise SystemExit(main())
