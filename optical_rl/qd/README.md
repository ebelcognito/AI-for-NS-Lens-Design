# QD (Quality-Diversity) starter notes

This folder is a **minimal** starting point for trying QD on the optical tasks in this repo.

## Why QD here?

PPO is good at finding **one** strong solution (a policy). QD methods like **MAP-Elites** try to find **many** strong solutions that are *different in a user-chosen way* (the “behavior descriptor”).

For optics design, that can be useful when you want:
- multiple high-power lens shapes with different *size/shape/curvature* constraints
- solutions that specialize for different detector locations / target rays
- a “library” of designs for downstream selection or fine-tuning

## What are the inputs/outputs?

For this project’s quad-curve glass setup, an introductory and practical choice is:
- **Genotype (input)**: a real-valued vector representing the **final glass shape** (corners + sags), e.g. 12 floats.
- **Evaluation output**:
  - **Fitness**: detector power (maximize)
  - **Descriptor (2D)**: any 2 numbers you want diversity over

Example descriptors you can start with:
- purely geometric: `(width, height)` of the glass bounding box
- optical behavior: `(irradiance centroid, irradiance spread)` along the detector (if available from the simulator)

## Code

- `optical_rl/qd/map_elites.py`: generic 2D MAP-Elites implementation
- `optical_rl/qd/quad_curve_glass_descriptors.py`: simple descriptor helpers
- `optical_rl/qd/demo_quad_curve_glass_mapelites.py`: quick demo that evaluates **design vectors** (requires `gymnasium` + Node runner)

## The most basic “is it working?” results

Without doing real science yet, you can sanity check:
- archive **coverage** increases above zero (it starts filling bins)
- **best_fitness** improves over random seeding
- visually inspect a few elites from very different bins to see if they are meaningfully different shapes

