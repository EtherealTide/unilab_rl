# Training metric schema

`uni_rl.logging.metric_schema` is the machine-readable source of truth for
TensorBoard and Weights & Biases scalars. Tags already emitted by upstream
RSL-RL are used verbatim. Fields RSL-RL does not have use the same terse
top-level groups (`Train`, `Loss`, `Policy`, `Episode`, and `Perf`) rather than
introducing algorithm- or runtime-specific namespaces.

Source metrics must already be canonical. There is no legacy translation layer;
unknown, retired, or mistagged fields fail closed. Fields assembled by a logger
from its dedicated telemetry inputs cannot be injected through the generic
learner metric map.

## Upstream alignment

The following RSL-RL tags are exact matches, including units and step semantics:

```text
Train/mean_reward
Train/mean_episode_length
Loss/surrogate
Loss/value
Loss/entropy
Loss/learning_rate
Policy/mean_std
Perf/total_fps
Perf/collection_time
Perf/learning_time
```

This registry covers source metrics emitted through `uni_rl` loggers. Direct
PPO keeps RSL-RL's native passthrough logger, so optional upstream fields such
as `Train/mean_reward/time`, RND diagnostics, and environment episode extras are
outside this registry rather than duplicated here.

`Train/mean_reward` is the mean of raw environment episode returns. Timeout
bootstrap corrections remain part of learner targets but are not included in
that displayed return.

Environment reward terms also follow RSL-RL's passthrough behavior:
`reward/<term>`. Async runners average all collector reports in a learner
iteration before persisting them. These are weighted reward rates, not
completed-episode returns. Reserved aggregate names such as `reward/mean` and
`reward/mean_ep100` fail closed.

## Step axis and schema fields

APPO and SAC-family writers use collector total environment steps, with runner
iteration as the pre-collector fallback. When that fallback is active, the
x-axis already is iteration and `Train/iteration` is omitted. `OnPolicyLogger`
and direct PPO use upstream RSL-RL's zero-based iteration axis. `log_interval`
only reduces write frequency; it does not change the axis or the contents of one
batched event, and the final iteration is always logged.

Every exact `MetricSpec` and the sole open `MetricFamilySpec` declare:

- canonical tag/prefix and owner;
- unit, local aggregation formula/window, and stale/mean semantics;
- independent DP reduction in `distributed_aggregation`;
- backend step axis and TensorBoard/W&B backend coverage;
- a description distinguishing nearby but unequal concepts.

Throughput and pipeline counts are cross-rank sums. Model statistics and phase
timings are cross-rank means. Episode returns and lengths are means of each
rank's latest 100-episode means; timeout rates and reward terms are means of
per-rank rates/report means. DP collective metrics are explicitly per rank. The
DP logger follows these declared reductions when it aggregates learner source
metrics.

## Extension fields

- SAC objectives use `Loss/actor`, `Loss/critic`, and `Loss/temperature`;
  they are not forced onto PPO's `Loss/surrogate` or `Loss/value`.
- SAC policy temperature is a post-update state and uses
  `Policy/temperature`; estimates and optimizer diagnostics use `Loss/entropy`
  and `Train/*_gradient_norm`. SAC's former pre-update action-standard-deviation
  chart is omitted instead of misusing the RSL-RL `Policy/mean_std` semantics.
- PPO/APPO update diagnostics use `Train/*` rather than a separate algorithm
  namespace. In particular, `Train/behavior_to_current_log_prob_delta` remains
  signed and is not called a KL.
- APPO runtime state stays under `Train/*`, matching RSL-RL's broad training
  group instead of a custom pipeline namespace.
- Additional timing fields stay under `Perf/*`. Millisecond fields end in `_ms`;
  the two upstream second fields retain their exact names.
- The learner-owned inference queue reports dynamic observation depth, action
  backlog, current/maximum in-flight requests, and publication lag under
  `Perf/collector_inference_*`. These are ordering diagnostics: a serial
  `observation[t] -> action[t] -> transition[t] -> observation[t+1]` dependency
  can legally use only one outstanding request even when the configured ring
  capacity is greater.
- GPU-resident replay reports configured ingress depth, live/high-water
  occupancy, cumulative producer backpressure wait, and early-return drops under
  `Train/replay_ingress_*`. These snapshots use host-shared sequence and counter
  metadata only and never synchronize or copy a device ingress tensor.

Learner main-thread phases are mutually exclusive. Nested inference and
env-step diagnostics are descriptions, not additional slices of the parent
phase. Residual learner time, display-only percentages, and cycle totals are
derived in-memory for the terminal and are not persisted as duplicate charts.
`Perf/collection_time` is the latest completed rollout measurement and is omitted
until that measurement exists; APPO currently supplies it, while the asynchronous
SAC-family collectors have no equivalent complete rollout phase to misrepresent.
`Perf/iteration_time` is emitted only when the
runner measured the complete iteration wall time; an accounted-phase sum is not
presented as wall time.

Deferred SAC/FlashSAC device reads use the same unsuffixed canonical
tags; their schema aggregation states that these learners sample the final update
when deferral is active. Historical migration notes below identify old fields
whose unsuffixed meaning was not declared.

Runner smoothing over ten collector reports is checkpoint state only. It is not
passed through the logger and has no TensorBoard/W&B field.

The audit retained only dynamic raw values that cannot be recovered from another
logged field or run configuration. Active collector throughput and learner replay
throughput can be derived from run configuration plus a measured iteration time;
the APPO update count follows its configured epoch/minibatch schedule. APPO staging-pool
occupancy is also omitted: active count only rises to its configured capacity and
does not describe training progress. The reward-normalization scale is omitted
when normalization is disabled rather than persisted as a constant value of one.
Nested timing diagnostics explain, but are not added to, their parent phase.

## Schema versioning and compatibility

Metric metadata and runtime manifests use separate version namespaces.

- TensorBoard/W&B scalar metadata is versioned by
  `METRIC_SCHEMA_VERSION = 1`. The current `METRIC_SPECS` registry and the
  `reward/` family are the v1 schema. Every summary produced by the current
  runner, including its minimal early-failure and cleanup-failure summaries,
  records this value as `metric_schema_version`; it is not emitted as a
  synthetic scalar.
- Runtime process/device/IPC/lifecycle state is versioned by
  `RUNTIME_MANIFEST_SCHEMA_VERSION = 1`. New run summaries record it as
  `runtime_manifest.schema_version`.

Runtime-manifest v1 freezes only fields already consumed by downstream tooling:
the top-level schema version, the public inference-ring capacity, collector
metric interval and backend device, inference-flight counters, replay-ingress
counters/timing, and the CUDA inference IPC event count when that budget is
present. Stable lifecycle sections are required for normal completed summaries;
an early failure summary may omit a section that never materialized.

The stable fields use these units and ownership semantics:

| Field | Producer | Unit / semantics |
|---|---|---|
| `inference_ring_capacity` | runner configuration | Slot count. |
| `collector_metrics_interval` | runner configuration | Collector control steps per report. |
| `collector_backend_device` | runner configuration | Non-empty accelerator device string, or `null` for a backend that needs no accelerator binding. It is required in completed manifests. |
| `inference_flight.queue_depth` | latest collector report | Instantaneous published-but-not-yet-consumed observation count. |
| `inference_flight.publication_lag` | latest collector report | Instantaneous in-flight publication lag. |
| `inference_flight.max_in_flight` | latest collector report | Maximum since the previous successfully published collector report; this is not a run-lifetime high-water. |
| `inference_flight.max_publication_lag` | latest collector report | Maximum since the previous successfully published collector report; this is not a run-lifetime high-water. |
| `replay_ingress.published_sequence` / `release_sequence` | replay ingress | Cumulative published / released batch count. |
| `replay_ingress.occupancy` | replay ingress | Instantaneous `published_sequence - release_sequence`. |
| `replay_ingress.high_water_occupancy` | replay ingress | Cumulative run high-water occupancy. |
| `replay_ingress.backpressure_waits` | replay ingress | Cumulative producer wait count. |
| `replay_ingress.backpressure_wait_s` | replay ingress | Cumulative producer wait time in seconds. |
| `replay_ingress.early_returns` and related return counters | replay ingress | Cumulative batch counts. |
| `inference_memory_budget.ipc_event_count` | inference budget preflight | CUDA IPC event count when the section exists. |

All other runtime-manifest fields are producer diagnostics. In particular,
`runtime_manifest.shutdown` is experimental and opaque in v1: its presence is
useful for diagnosis, but consumers must not require or interpret its internal
fields until that section is explicitly promoted to a stable contract.

Within a schema major version, new optional fields may be added together with
schema, test, and documentation updates. A stable field may not be renamed,
removed, retyped, or changed in unit or null semantics without incrementing the
schema version. Unsupported versions fail closed. The historical TensorBoard
migration table below remains a narrowly scoped reader for old runs; it is not
a general compatibility layer for future metrics.

## Old-to-new reference

Historical event files are immutable. This table is only for reading old runs;
new runs do not emit or accept the old keys.

| Old field | New field |
|---|---|
| `iteration` | `Train/iteration` |
| `reward/mean` or `reward/mean_ep100` | `Train/mean_reward` |
| `episode/length` | `Train/mean_episode_length` |
| `episode/timeout_rate` | `Episode/timeout_rate` |
| `reward/<term>` | `reward/<term>` (semantic contract now explicit) |
| `Loss/surrogate`, `loss/policy_loss`, `train/surrogate_loss`, `train/actor_loss` | `Loss/surrogate` for PPO/APPO; SAC uses `Loss/actor` |
| `Loss/value`, `loss/value_loss`, `train/value_loss` | `Loss/value` |
| `Loss/entropy`, `train/entropy`, `train/policy_entropy`, `train/actor_entropy` | `Loss/entropy` |
| `train/qf_loss`, `train/critic_loss` | `Loss/critic` |
| `train/alpha_loss`, `train/temperature_loss` | `Loss/temperature` |
| `train/alpha`, `train/temperature` | `Policy/temperature` |
| `Policy/mean_std`, `policy/mean_std` | `Policy/mean_std` |
| `train/action_std` | _removed_; SAC sampled it before the actor update |
| `Loss/learning_rate`, `optim/learning_rate` | `Loss/learning_rate` |
| `grad/global_norm` | `Train/global_gradient_norm` |
| `train/actor_grad_norm` | `Train/actor_gradient_norm` |
| `train/critic_grad_norm` | `Train/critic_gradient_norm` |
| `train/kl`, `ppo/approx_kl` | `Train/approx_kl` |
| `policy_kl/behavior_to_current_kl` | `Train/behavior_to_current_log_prob_delta` |
| `vtrace/rho_clip_fraction` | `Train/vtrace_rho_clip_fraction` |
| `vtrace/rho_raw_p99` | `Train/vtrace_rho_p99` |
| `appo/updates_executed` | _removed_; derive from the run configuration |
| `target_q_max`, `train/target_q_max` | `Train/target_q_max` |
| `target_q_min`, `train/target_q_min` | `Train/target_q_min` |
| `reward_scale_std`, `train/reward_scale_std` | `Train/reward_scale_std` |
| `train/staging_pool_len`, `train/staging_pool_capacity` | _removed_; capacity comes from run configuration and occupancy is not training progress |
| `train/available_on_arrive` | `Train/ring_available_slots` |
| `train/rollouts_read` | `Train/rollouts_read` |
| `train/dp_sync_time` | `Perf/dp_gradient_sync_ms_per_rank` |
| `train/dp_gradient_sync_calls` | `Perf/dp_gradient_sync_calls_per_rank` |
| `Perf/total_fps`, `perf/steps_per_sec` | `Perf/total_fps` |
| `perf/collector_active_steps_per_sec` | _removed_; derive it from collector timing and run configuration |
| `perf/effective_samples_per_sec`, `perf/learner_samples_per_sec` | _removed_; derive replay rows from run configuration and `Perf/iteration_time` |
| `perf/iter_ms` | `Perf/iteration_time` |
| `perf/collect_time_ms`, `timing/collector_rollout_ms` | `Perf/collection_time` |
| `perf/train_time_ms`, `timing/learner_train_ms` | `Perf/learning_time` |
| `perf/learner_*_pct`, `perf/collector_cycle_ms` | _removed_; derive from canonical timing fields |
| `timing/learner_other_ms` | _removed_; derive from iteration and mutually exclusive learner phases |
| `timing/<phase>_ms` | `Perf/<phase>_ms` |
| `timing/collector_<phase>_ms` | `Perf/collector_<phase>_ms` |
