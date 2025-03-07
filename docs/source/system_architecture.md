# ⚙️ System Architecture

ICLand is a simulation framework built on [MuJoCo-XLA (MJX)](https://mujoco.readthedocs.io/en/stable/mjx.html) that integrates procedural world generation, agent-based interaction, and parallelised simulation updates. The system is structured into [initialisation](#initialisation), [simulation](#simulation-loop), and [parallelisation](#parallelisation) components.

![System Architecture Diagram](_static/system_design.jpeg)

## Initialisation

1. **Configuration Module (`icland.config`)**
   - Uses the [MuJoCo (CPU) API](https://mujoco.readthedocs.io/en/stable/overview.html) to generate a Base MJX Model.
   - Produces ICLandConfig, which defines simulation parameters.

2. **Sampling Module (`icland.sample`)**
   - Uses Random Keys ([`jax.random.PRNGKey`](https://docs.jax.dev/en/latest/_autosummary/jax.random.key.html)) and the Wave Function Collapse Engine (`sample_world`).
   - Generates a world tilemap and forms ICLandParams.

3. **Initialisation Module (`icland.init`)**
   - Modifies the Base MJX Model into an Edited MJX Model.
   - Creates an ICLandState initialised with Initial MJX Data.

## Simulation Loop

1. **Simulation Engine (`icland.step`)**
   - Updates Current MJX Data using [MuJoCo-XLA](https://mujoco.readthedocs.io/en/stable/mjx.html) (`mjx.step`).
   - Calls the Rendering Engine (`render`) to generate frames of agent perspectives.
   - Produces observations and rewards for the agent.

2. **Agent/Policy Neural Network**
   - Processes observations and rewards.
   - Generates actions, which are fed back into the Simulation Engine.

## Parallelisation

- Batched operations enable the simultaneous processing of multiple:
  - ICLandParams
  - ICLandStates
  - Actions/Observations
- JAX vectorisation supports efficient execution.

## External Dependencies

- [MuJoCo API](https://mujoco.readthedocs.io/en/stable/overview.html) for physics simulation.
- Random Keys ([`jax.random.PRNGKey`](https://docs.jax.dev/en/latest/_autosummary/jax.random.key.html)) for procedural generation.
- Reward Function for training and evaluation.
