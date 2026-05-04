# `model/skel_blind/` — Skeleton-blind motion representation

Infrastructure for the ICML 2028 oral submission's §2 method. See the spec at
`docs/superpowers/specs/2026-04-16-cross-skel-oral-design.md`.

## What this package provides (Plan A Foundation)

- `slot_vocab` — 32-slot vocabulary with fixed indices.
- `contact_groups_canonical.load_canonical_contact_groups` — translates
  the heterogeneous keys in `eval/quotient_assets/contact_groups.json`
  (30+ raw labels: `L`, `R`, `L1..4`, `L_arm`, `L_mid*`, `tail`, `front`,
  `mid_back`, `all`, ...) into the canonical 32-slot vocabulary at load
  time. Three previously-`_unresolved` skeletons (Pigeon, Tukan, Pirrana)
  get explicit per-skeleton overrides authored from joint-name evidence.
  The source JSON is preserved verbatim (audit trail).
- `slot_assign.assign_joints_to_slots` — deterministic joint→slot mapping
  using the canonical loader plus a bilateral-depth heuristic for
  `mid_leg_*` residuals on uncovered joints. Skeleton lookup uses
  `cond['object_type']`.
- `encoder.encode_motion_to_invariant` — `[T, J, 13] → [T, 32, 8]`, where
  the 8 channels are `[pos (3), contact (1), vel (3), phase (1)]`. Phase
  is strike-to-strike linear interpolation between contact rising edges
  (gait-cycle convention), with global linear-ramp fallback when fewer
  than two strikes are detected.
- `fk_decoder.fk_decode` — baseline pass-through decoder (writes slot
  positions/velocities/contacts back onto their assigned joints). Not
  differentiable IK; see Plan C.

## What this package does NOT provide yet

- Differentiable Gauss–Newton IK layer (Plan C).
- Target-predictive CFM generator (Plan C).
- V_eq equivariant variant (Plan D).
- Paired benchmark data (Plan B).
- Training loop integration with AnyTop (Plan C).

## Stability guarantees

- Slot indices 0..31 are frozen. Renumbering requires an explicit
  migration because pre-registered benchmarks reference these indices.
- `encode_motion_to_invariant` output shape is `[T, SLOT_COUNT=32, CHANNEL_COUNT=8]`.
- Channel layout within each slot: position 0:3, contact 3:4, velocity 4:7, phase 7:8.

## Testing

```
conda activate anytop
pytest tests/skel_blind/ -v
```

15 test files cover all components. The integration test
`test_integration_e2e.py` runs end-to-end on 5 topology-diverse
representative skeletons.
