================================================================================
PARTIAL OPTIMIZATION (ENGLISH) - Simplified model up to the medial phalanx
================================================================================

PURPOSE
--------------------------------------------------------------------------------
This folder contains the English version of the SIMPLIFIED optimization: a fair,
symmetric comparison between two metaheuristics, Differential Evolution (DE) and
a Genetic Algorithm (GA), on the simplified finger-exoskeleton model. The
simplified model has 16 parameters and covers the kinematics up to the medial
phalanx (MCP and PIP/IFP joints). Both optimizers minimize the SAME objective
function and tune their hyperparameters with Optuna, so the DE vs GA comparison
measures the algorithm, not the problem formulation.

The objective function combines:
  - Joint-weighted Chamfer distance (IFP weight 0.40, IFD weight 0.60), which
    measures the shape similarity between the exoskeleton trajectory and the
    motion-capture trajectory.
  - Optimal rigid alignment (Kabsch/Procrustes) before measuring the error.
  - Monotonicity penalty (W_MONO = 5.0) so the motion advances coherently with
    no artificial backtracking.
  - Soft regularization of dimensions and auxiliary variables.
  - Fixed penalty of 1000 for mechanisms that fail to assemble.

CONTENT AND WHAT EACH SCRIPT DOES
--------------------------------------------------------------------------------
common.py
    Shared utilities: motion-capture loading (load_mocap), Chamfer distance
    (chamfer_distance), optimal rigid alignment (optimal_rigid_transform /
    apply_transform), monotonicity penalty (monotonicity_penalty), mechanism
    solvers (sol_5_barras, solve_four_bar, circle_intersections) and the genetic
    algorithm.

simplified_model.py
    Defines the 16-parameter model: bounds, NAMES, fitness_function(p) (returns
    1000 if the mechanism does not assemble), run_kinematics, evaluation metrics
    (in mm) and kinematic validation. Loads the mocap CSV that sits in this same
    folder.

optimize_simplified_DE.py
    Optimizes the simplified model with Differential Evolution. Two-stage flow:
    first Optuna/TPE (25 trials, seed 42) tunes the differential_evolution
    hyperparameters (strategy, popsize, mutation, recombination), then it runs
    the deep optimization.

optimize_simplified_GA.py
    Same, but with the Genetic Algorithm, using the same Optuna/TPE scheme (25
    trials, seed 42) so the comparison is symmetric.

mocap_pinza_fina_120pts.csv
    Fine-pinch grasp motion-capture data (120 points). It must be in this folder
    because the model module loads it by relative path.

requirements.txt
    Exact dependency versions.

DEPENDENCIES / EXACT VERSIONS
--------------------------------------------------------------------------------
Python 3.9 or 3.11 (tested on both).
  numpy      == 2.0.2
  scipy      == 1.13.1
  pandas     == 2.3.3
  matplotlib == 3.9.4
  optuna     == 4.9.0

Installation:
  pip install -r requirements.txt

HOW TO RUN AND IN WHICH ORDER
--------------------------------------------------------------------------------
Run the commands from INSIDE this folder (the CSV path is relative).

  1) python3 optimize_simplified_DE.py     (DE optimizer, slow)
  2) python3 optimize_simplified_GA.py     (GA optimizer, slow)

Both scripts write their own result files. There is no separate comparison
script in this English folder; run each optimizer and read its outputs.

EXPECTED RESULTS (known numbers, seed 42)
--------------------------------------------------------------------------------
Differential Evolution (DE):
  global error = 2.17 mm   (IFP error 2.82 mm, IFD error 1.53 mm)
  fitness = 0.002335       nfev = 135970

Genetic Algorithm (GA):
  global error = 2.55 mm   (IFP error 3.68 mm, IFD error 1.43 mm)
  fitness = 0.00272        nfev = 15052

Conclusion: DE wins on global error (2.17 mm vs 2.55 mm), so it is the
metaheuristic later used for the complete model.

REPRODUCIBILITY NOTE
--------------------------------------------------------------------------------
Even with seed 42, the final numbers depend on the library VERSIONS. Optuna (TPE
sampler) selects the hyperparameters, so a different version selects different
hyperparameters for the same seed; SciPy fixes the differential_evolution
trajectory. Use exactly the versions in requirements.txt to reproduce the
numbers above.
