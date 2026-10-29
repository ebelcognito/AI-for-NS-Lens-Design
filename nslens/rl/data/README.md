# RL/SFT dataset contracts

This directory is the data boundary between the two learned policies and the two optical
engines. Generated datasets belong under runs/rl_datasets/ and are ignored by git; the
builders, validators, and schema examples are source.

See [TRAINING_PLAN.md](TRAINING_PLAN.md) for the staged Model-A/Model-B recipe, the remote
CODE V pilot, evaluation criteria, and the representation needed for genuinely novel
topology rather than archive imitation.

The current local build produces three complementary supervision layers. They should not
be concatenated blindly: each answers a different question and has a different split unit.

## Model A: topology/starting-point proposer

Build the current local corpus without CODE V:

    python -m nslens.rl.data.build_proposer \
      --certified runs/rl_tasks/certified_20k.json \
      --out runs/rl_datasets/proposer_v1 \
      --n-format 2000 \
      --augment-per-cert 8 --workers 20

This creates three corpora, kept separate on purpose:

- format: uniform-random geometry that passes the cheap buildability gates. It contains
  no topology or performance labels and is the thin scaffold for topology-blind RL.
- directed: a requested TIR-face multiset and a hardened certified design that reaches it.
  This is the operational known-topology seed generator.
- unconditioned: the certified design with topology removed from the prompt. This is a
  multimodal archive prior. Because targets came from the archive, it is archive
  distillation and must not be described as topology emergence.

Perturbation augmentation is traced locally after four-decimal serialization. A candidate
is retained only if it preserves the requested topology and thickness cell, meets 80% of
the certified score, and passes clean packing, exit convergence, and critical-margin gates.

test_cell is held out by complete topology class. validation_seen deliberately shares
classes with train and is only for loss/early-stopping; it is not an OOD result.

The generated ``sample_weight`` gives every certified spec lineage unit total mass. A
surviving basin with eight perturbations therefore does not count nine times more than a
narrow basin with only its anchor. The audit also reports physical anchor lineages because
one warm-start elite can certify several FOV/window prompts.

Validate generated files:

    python -m nslens.rl.data.validate proposer \
      runs/rl_datasets/proposer_v1/directed/train.jsonl

Audit exact leakage and normalized nearest-neighbour redundancy:

    python -m nslens.rl.data.audit runs/rl_datasets/proposer_v1 \
      --out runs/rl_datasets/proposer_v1/audit.json

## Model B pretraining: local search actions

The QD traces already contain state/action/outcome supervision: each consecutive ten rows
are the candidates from one CMA-ME ``ask()`` call around one parent. Build compact action
groups and chosen/rejected pairs with:

    python -m nslens.rl.data.build_search_actions --runs runs \
      --out runs/rl_datasets/search_actions_v1 --cap-per-tier 5000

The builder reads the experiment label and parameter names from each run's ``archive.json``.
For v1 it accepts only registered, normalized-search, relative-sag traces with the current
nine-parameter fold layout. Absolute-sag vectors, ten-gene surfaces, prism layouts, missing
metadata, and unregistered experiments are excluded with a reason in the manifest.

Ranking is only within one same-parent population and exactly matches the optimizer:
``new > improved > none > invalid``; fitness or improvement delta breaks ties inside a
tier. No historical scores are compared across runs. Splits hold out complete random seeds,
including across run families, to prevent repeated CMA random directions leaking.

Outputs:

- ``groups/``: all ten candidates, exact normalized moves, archive outcomes, and ranking.
- ``preferences/``: the best action against a hard negative and, where available, the
  nearest invalid sibling. These are suitable for DPO/ranking or best-action SFT.

Historical action status came from the gate stack active in each source run. Preserve it as
search-history supervision, but derive the physically hardened refinement view before SFT:

    python -m nslens.rl.data.harden_search_actions \
      --input runs/rl_datasets/search_actions_v1 \
      --out runs/rl_datasets/search_actions_hardened_v1 --workers 20

This re-scores every parent/candidate using one canonical configuration. SFT rows require
non-intersecting geometry (including 12 mm detector and 2 mm source clearance), the 0.268
sag-ratio cap, in-class topology/thinness, all-field clean packing, exit imaging, critical
margin, topology preservation, and improvement when the parent already passes. Near-misses
remain in preference groups but cannot become positive SFT targets.

Validate them with ``search-group`` and ``search-preference`` dataset kinds.
The strict SFT view uses the ``hardened-action-sft`` validator.

Audit effective state count, label composition, preference strength, action coverage,
same-prompt conflicts, bound saturation, split leakage, and rough token volume:

    python -m nslens.rl.data.audit_hardened_actions \
      runs/rl_datasets/search_actions_hardened_v1 \
      --out runs/rl_datasets/search_actions_hardened_v1/audit.json

In this corpus, ``hard_pass`` means physically acceptable under every configured gate; it
does not impose a minimum optical fitness. Keep feasibility-repair SFT separate from
strictly fitness-improving SFT when deriving a LoRA training view.

## Canonical local quality catalogue

The old trace fitnesses came from different field sets and gate stacks, so they are not
safe absolute labels. This builder uses old labels only to choose a topology-diverse set,
then ray-traces every candidate again under one fingerprinted nine-field hardened config:

    python -m nslens.rl.data.build_hardened_catalogue --runs runs \
      --out runs/rl_datasets/hardened_catalogue_v1 --workers 20

All verdict stages are retained. Only designs with ``stage=scored``, fitness at least 0.05,
and ``clean_all``, ``exit_ok_all``, and ``margin_ok`` become training-positive records.
Positive splits hold out entire realized topology classes; rejected/low-quality records are
kept separately for discriminators, reward models, and hard-negative sampling. This is a
local 2D optical label and must remain distinct from future CODE V validation.

## Model B: CODE V refinement controller

No CODE V data can be generated on this machine. The remote COM harness should append one
nslens.codev_transition.v1 record for every evaluated perturbation, including rejected
and failed candidates. Start from codev_transition.example.json.

Load-bearing conventions:

1. parameter_names, parameters, parameter_scales, active_mask, and normalized_delta have
   identical order and length.
2. The physical action is parameters + normalized_delta * parameter_scales. Bounds are
   enforced by the harness and clipping is recorded, never hidden.
3. CODE V merit is assumed to be minimized, so delta_merit = merit_before - merit_after.
   Positive is improvement.
4. Log every candidate from the same state with a shared (episode_id, step) and distinct
   candidate_index. Those groups become preference or GRPO groups without reconstruction.
5. Preserve the full metric and constraint vectors. accepted is a policy decision, not a
   sufficient learning label; exact deltas remain usable when the rule changes.
6. Version CODE V, the merit function, and the JSON-to-CODE V translator on every record.
7. Keep large .seq, plots, and reports as artifacts referenced by hash/path. Do not copy
   them into the model prompt or JSONL record.

Validate a future export:

    python -m nslens.rl.data.validate codev-transition transitions.jsonl

From these records we can derive four later datasets without another CODE V run:
successful-action SFT, same-state action preferences, state/action-to-delta surrogate
regression, and full offline-RL trajectories.

Once transitions are copied back, derive the first three views with:

    python -m nslens.rl.data.derive_codev transitions.jsonl \
      --out runs/rl_datasets/codev_v1

The derivation holds out complete ``design_id`` values and ranks candidates by feasible,
then traceable, then merit improvement. It deliberately does not learn the harness's
``accepted`` decision as ground truth. The raw transition log remains the source of truth.

For token efficiency, the refinement policy should consume the numeric state and emit only
``normalized_delta`` (plus an optional trust scale). It should not receive full CODE V
reports or narrate each iteration. Keep reports as artifacts; feed the policy fixed-size
metrics, constraint slacks, recent accepted deltas, and a small candidate budget. Use the
large topology model only for initial designs and occasional macro-level restarts.
