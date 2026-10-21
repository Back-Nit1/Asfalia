# Specialist request disposition

These files are historical handoffs from the parallel implementation branches. The release lead reviewed every item;
the source of truth is the merged code and current documentation.

| Request | Disposition |
| --- | --- |
| Agent C | CLI help delegation, full world-config propagation, experiment command, and slow-marker registration resolved |
| Agent D | Durable backend contract, metrics/checkpoint ingestion, real local frozen evaluator, and replay export resolved; real debug-fork counterfactual evaluation remains deferred |
| Agent E | Cloud plan/reconcile CLI and persistent local registry resolved; pinned league/rollout versions are validated but not yet consumed by a rollout-only worker |
| Agent F | Trajectory/report/ablation CLI and training-stratum guards resolved; four engine-changing diagnostic ablations remain explicitly unavailable |
| Agent G | Inference CLI and optional inference dependency extra resolved |
| Agent H | Daytona SDK pin and removal of direct dotenv use resolved; rollout-only job execution remains deferred |

Deferred items are listed again in `docs/release_acceptance.md`; commands refuse unsupported modes instead of producing
placeholder results.
