# Daytona backend and synthetic-service testbed (Agent H)

Status legend: **FAKE** = tested against an in-process fake provider; **LOCAL** = run on this machine with real code, no
cloud; **LIVE** = ran on Daytona; **UNEXECUTED** = code exists, never run. **No LIVE result exists for anything below.**

## What changed
- `infra/daytona_backend.py`: `DaytonaBackend` = `GatewayBackend("daytona", DaytonaGateway, Registry, caps)`: the same
  lifecycle as local/Modal (bundle validation, caps before launch, idempotent submit, durable attempt records, drain /
  checkpoint-and-stop via control words, force-cancel, checkpoint freshness, resume lineage, reconcile). Only `SdkClient`
  imports the SDK; pinned `daytona==0.225.0` (signatures read from that release; `SdkClient` refuses another version).
- `infra/cloud_school.py`: the second Daytona lifecycle (thread pool, tarball upload, f-string shell, `.env` loading) is
  gone; `daytona_main` builds the same `school_specs` and calls the common backend.
- Credentials: the SDK reads `DAYTONA_API_KEY` from the process environment. Nothing reads `.env`; values are never
  printed (tests grep the code and run the smoke script with a dummy key). The sandbox job receives no provider credentials.
- Commands: only argv lists (`shlex.join`) with validated ids (`SAFE_ID`), the job spec travels as an uploaded file, and the
  setup string is constant (a test asserts it).
- Config: `configs/cloud_daytona.yaml` (real hospital factory, explicit caps, pinned snapshot slot, idle stop 15 min,
  provider TTL 120 min). `cloud-plan --backend daytona` is now a real dry run: resolves + validates the bundle, shows caps and
  a labelled *estimate*, checks the credential by name only, never imports the SDK or contacts Daytona (tested).

## Lifecycle semantics (Daytona-specific)
| Concern | Behaviour |
|---|---|
| Ownership | sandbox labels `owner=asfalia`, `asfalia.run`, `asfalia.attempt`; control/cleanup only touch these |
| Durable ids | `remote_id = sandbox|start-epoch|run|attempt`; restart recovery needs only the registry + provider (session command list) |
| Timeout | controller hard-kills at `timeout_s + grace_s + 1800` (setup budget); job soft-stops itself at `timeout_s` |
| Retry | bounded by `max_retries`; re-runs the job command in the same sandbox, resumes from the attempt's own `latest.pt` |
| Transfer | artifacts pulled sandbox -> `mirror_dir`, size-checked per file and manifest-hash-verified; sandbox deleted **only after** that |
| Partial upload / mismatch | job `FAILED("...sandbox kept")`, nothing deleted |
| Failure / cancel / timeout | partial checkpoint mirrored first, then the sandbox is deleted; resume via a new attempt (`resume_from_attempt` uploads the previous mirror) |
| Orphans | `list_attempts` includes owner-labelled sandboxes unknown to the registry; `reconcile` reports them, never deletes |
| Idle / TTL | provider-side `auto_stop_interval` and `auto_delete_interval` are set on creation |

## Smokes
- Fake-provider: `PYTHONPATH=src pytest tests/infra/test_daytona.py tests/infra/test_daytona_safety.py` (FAKE).
- Workspace smoke (`infra/daytona_workspace.py`): `git archive` of a pinned full sha is uploaded (no git credentials, no
  clone), controlled deps installed (runtime deps + pytest), one `tests/<file>.py` run via fixed argv, report downloaded and
  re-hashed, commit + archive sha + report sha returned, sandbox always deleted. Only its logic is FAKE-tested; the actual
  in-sandbox pytest run is **UNEXECUTED**.
- Graph job smoke: a tiny `kind="train"` hospital job via `DaytonaBackend`. The same spec ran **LOCAL** (`scripts/daytona_smoke.py --local`:
  state `done`, 64 env steps) and under the **FAKE** provider. There is no separate rollout-only job kind yet (see requests),
  so "rollout smoke" here means this training attempt with the same resolved bundle; no per-step cross-cloud RPC.
- Capped live smoke: `ASFALIA_DAYTONA_SMOKE=1 python scripts/daytona_smoke.py --live --yes --max-minutes 5 --max-cpu 1 --max-steps 64`.
  Without all of those (and `DAYTONA_API_KEY` present) it prints the caps/estimate/setup help and runs nothing live
  (`--local` runs the local fallback). **UNEXECUTED: no authorization or hard caps were given for credentialed work.**

## Synthetic-service testbed (`src/hospital_rl/testbed/`, optional, LOCAL)
Invented `SYN-` ids only (no names/DOB/address columns; tested), <= 200 rows per node, deterministic per seed with exact
`reset`, benign deterministic workload generator, mock identity + imaging-metadata tables. Policies get `ActorAPI` with
`act(type, node)` and `observe()` only; `act` accepts only the graph env's attacker/defender action vocabulary against known
nodes (no SQL, no shell; injection strings are rejected). `AuditLog` is a separate database the actor API never holds.
`IsolationPolicy` encodes internal-only networking (egress denied except 3 internal hosts) and strips provider credentials
from a policy's environment. Honest limits: sqlite stands in for PostgreSQL (schema is plain SQL; **no PostgreSQL service,
container network or resource limits were built or run**); Python attribute privacy is not a security boundary: real isolation
would be the process/container boundary the deployment bundle must provide (UNEXECUTED).

Adapter (`testbed/adapter.py`): replays a graph episode's action sequence into the testbed and diffs node flags. Real result on
train seeds 3, 4, 5 (8 nodes, horizon 20, scripted attacker vs greedy defender): 2 mismatches each, both `logging`. Cause checked
for seed 3: the graph exhausted its budget and emitted 2 `control_lapsed` events (unfunded recurring logging reverts to baseline);
the testbed has no budget model. Defender jobs with duration > 1 are reported as `pending_job`, not hidden. This tests adapter
semantics only; it says nothing about real-hospital validity, HIPAA, or remediation effectiveness.

## Not done
Live Daytona anything; a prebuilt pinned snapshot (config slot exists, `snapshot: null`); PostgreSQL/container deployment of the
testbed; a rollout-only job kind; network allow-listing on the sandbox (`network_block_all` is not set because setup pip-installs).
