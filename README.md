# Green Quantum Data Center (GQDC) — Project Skeleton

This repository is dedicated to the research project **"Energy/Carbon-Aware Scheduling and Control for Quantum Workloads."**

**Goal:** Reduce **energy per job by $\ge$ 20%** without significant increases in SLA violations, providing fully reproducible code.

-----

## Execution (Minimal)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-min.txt
python -m gqdc.experiments.run_ablation --config configs/config.yaml
```

## Execution (Full with MILP/RL)

```bash
pip install -r requirements.txt
# MILP Example (after completing milp.py)
# python -m gqdc.experiments.evaluate --scheduler milp --config configs/config.yaml

# PPO Training (after improving env and train)
# python -m gqdc.rl.train_ppo --config configs/config.yaml
```

-----

## Project Structure

The `gqdc/` core package consists of the following submodules:

  * **`simulate`**: Environments for data center and quantum hardware simulation.
  * **`quantum`**: Quantum circuit models and noise/power profiles.
  * **`scheduler`**: Logic for job placement and carbon-aware queue management.
  * **`rl`**: Reinforcement Learning agents (e.g., PPO) for dynamic control.
  * **`control`**: Power management and cooling system controllers.
  * **`metrics`**: Calculations for PUE (Power Usage Effectiveness), CUE (Carbon Usage Effectiveness), and SLA adherence.
  * **`experiments`**: Scripts for ablation studies and benchmarking.

### Other Directories

  * **`configs/`**: YAML configuration files for different scenarios.
  * **`docs/`**: Documentation (including the North Star Framework).
  * **`requirements*.txt`**: Dependency manifests.
  * **`LICENSE` & `.gitignore`**: Project metadata and version control exclusions.

-----

**Generation Date:** 2025-10-26 22:41
