# Folded-optics learning pipeline

## Architecture and boundary

The useful architecture is hierarchical:

1. **Topology/seed policy:** specification -> several candidate ray-path intents and
   buildable JSON starting points.
2. **Fast local verifier:** JSON -> geometry verdict, traced path, topology, field metrics,
   gate verdicts, and scalar reward.
3. **Refinement policy:** compact numeric state -> a batch of normalized parameter moves.
4. **CODE V authority:** translated JSON -> high-fidelity merit/constraints; its measured
   transitions replace local proxy labels for final refinement.

The topology policy can reasonably be an LLM because it handles a variable structured
specification and multimodal outputs. The current refinement problem is nine continuous
numbers. A small numeric policy plus a surrogate/ranker is the better default there; an LLM
should only be retained if the remote JSON has variable numbers/types of surfaces. In either
case the action output is a short normalized vector, never an optimization narrative.

## What exists locally now

| Corpus | Supervision | Honest use |
|---|---|---|
| `proposer_v1/format` | buildable JSON, no optical label | output grammar and cheap geometry scaffold |
| `proposer_v1/directed` | spec + requested known topology -> hardened certified seed | known-class topology-conditioned SFT |
| `proposer_v1/unconditioned` | spec without topology -> archive seed | archive prior; sample diversity, not proof of novelty |
| `search_actions_v1/groups` | one parent + ten applied moves + exact historical QD ranking | search-history/action-surrogate pretraining |
| `search_actions_hardened_v1` | every parent/action re-scored through current hard gates | refinement SFT and safety-first preferences |
| `hardened_catalogue_v1` | one canonical local re-score for positives and negatives | reward/discriminator training and held-topology evaluation |
| raw `codev_transition.v1` | remote state/action/result for every attempted move | source of truth for the final refiner |

Record count is never reported as independent sample count. Proposer perturbations are
weighted by certified-spec lineage; search groups are weighted by distinct parent state.

## Topology policy training order

1. Train format compliance briefly on `format`; stop once exact parse/buildability is high.
2. Train the directed task using `sample_weight`. Validate on `validation_seen`; make all
   model and hyperparameter decisions before looking at `test_cell`.
3. Run the unconditioned archive prior as a separate adapter/experiment. Generate multiple
   samples per spec and score coverage; do not merge its metrics with directed topology hit.
4. Only then run local online RL. For each task, sample 8-16 designs, batch them through the
   reward service, and use the staged reward. Resample tasks into the 20-80% pass-rate band
   to prevent zero-advantage groups.

Primary Model-A metrics are exact JSON rate, buildable rate, requested topology hit,
hardened-pass rate, best-of-N fitness, number of unique full signatures, and performance on
topology-held-out `test_cell`. Token loss alone is not an optical metric.

### What “novel topology” requires

SFT on archive designs can reproduce and recombine observed modes; it cannot establish
novelty. The current coarse label is a TIR-face multiset and the full signature is measured
after tracing. That supports novelty **within the current four-face, fixed-length element**.

For arbitrary new fold structures, add an explicit variable-length topology DSL/graph, for
example ordered interaction tokens with face/surface identity and T/TIR/R type. The policy
then proposes `(topology graph, geometry)`, while the verifier labels intended-vs-realized
topology. Useful additional records are:

- spec -> several distinct feasible topology graphs;
- topology graph -> one or more geometry seeds;
- intended graph + geometry -> realized graph + mismatch edit distance;
- topology mutation -> feasibility, quality change, and whether it created a new signature;
- equal-spec preference pairs balancing optical quality against catalogue novelty.

Novelty reward must be measured against a frozen training catalogue and reported separately
from quality. Otherwise the policy can earn novelty by producing broken or noisy paths.

## Refinement training order

1. Use `search_actions_hardened_v1/sft`, not historical winners, as the local action SFT
   initialization. It teaches scales, clipping, current validity boundaries, topology
   preservation, and locally improving directions—not CODE V merit.
2. Train a state/action -> outcome surrogate on every CODE V transition, including failures.
   Predict merit delta, trace success, individual constraint slacks, topology preservation,
   and per-metric changes rather than only accepted/rejected.
3. Train same-state preferences. Rank feasible above infeasible, then traceable above failed,
   then higher merit improvement. Never treat the harness's old `accepted` decision as the
   sole label.
4. Fine-tune the action policy on successful feasible moves, then preference-optimize it.
5. Evaluate online at an equal CODE V call budget against the existing random perturbation
   harness, CMA-ES, and a surrogate-assisted optimizer. Report merit gain per CODE V call,
   wall time, violation rate, topology-preservation rate, and tokens per accepted gain.

The derivation command is:

```bash
python -m nslens.rl.data.derive_codev transitions.jsonl \
  --out runs/rl_datasets/codev_v1
```

## Remote CODE V pilot

Start with a diagnostic pilot, not a giant generation run:

- 25 baseline designs stratified by topology, quality, thinness, and failure mode;
- 2 independent episodes per design;
- 15 same-state decision steps per episode;
- 8 candidate actions at each step;
- about 6,000 CODE V evaluations total.

At each state mix candidate sources: random log-spaced perturbation scales, coordinate moves,
CMA/surrogate proposals, the local pretrained policy, and later the CODE V-tuned policy.
Logging only policy actions creates a narrow behavior dataset and prevents a fair comparison.

After the pilot, inspect feasible/improving yield, constraint-label completeness, action-scale
coverage, repeated-state redundancy, and held-design performance. Expand toward roughly
40,000 evaluations only if those diagnostics show learning signal. Every candidate from a
state must share `(episode_id, step)`, have a distinct `candidate_index`, and preserve exact
merit/translator/CODE V version identifiers.

## Stop conditions

Do not claim success from lower training loss or a few attractive designs. Advance only if:

- Model A improves hardened best-of-N topology hit/quality at equal engine calls;
- Model B improves CODE V merit per call on held-out `design_id` values;
- gains persist after constraints and topology preservation are enforced;
- token use falls because optimization steps are numeric and compact;
- novelty is a verified new realized signature, not merely new JSON text.
