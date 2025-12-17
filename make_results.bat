
@echo off
REM One-click pipeline to produce full paper artifacts
set CFG=configs\config.yaml

echo === Facility CI ===
python -m gqdc.experiments.compare_facility_ci --config %CFG% --seeds 8 --fixed_setpoint 6.01 --cool_fraction 1.09 --cop_min_at6 2.32 --cop_max_at12 5.66 --ambient_base_c 21.1 --ambient_swing_c 7.11 --econ_max_gain 0.21

echo === Bootstrap significance ===
python -m gqdc.experiments.bootstrap_tests --config %CFG% --seeds 8 --boots 4000 --fixed_setpoint 6.01 --cool_fraction 1.09 --cop_min_at6 2.32 --cop_max_at12 5.66 --ambient_base_c 21.1 --ambient_swing_c 7.11 --econ_max_gain 0.21

echo === Emissions (aggressive flexible) ===
python -m gqdc.experiments.compare_emissions_aggressive --config %CFG% --carbon_base 650 --carbon_swing 400 --threshold 480 --forecast 60 --drop_min 40 --deferral_step_min 4 --max_deferrals 8 --deadline_s 5400 --seeds 5

echo === Pareto sweep ===
python -m gqdc.experiments.pareto_sweep --config %CFG% --seeds 5 --carbon_base 650 --carbon_swing 400 --steps 2,3,4,5 --maxdefs 2,4,6,8 --drops 20,30,40,60 --forecasts 30,45,60,90 --fixed_setpoint 6.01 --cool_fraction 1.09 --cop_min_at6 2.32 --cop_max_at12 5.66 --ambient_base_c 21.1 --ambient_swing_c 7.11 --econ_max_gain 0.21

echo === COP ablation (nonlinear) ===
python -m gqdc.experiments.cop_nonlinear_ablation --config %CFG% --seeds 5 --fixed_setpoint 6.01 --cool_fraction 1.1 --cop_min_at6 2.4 --cop_max_at12 5.6 --ambient_base_c 22.0 --ambient_swing_c 6.5 --econ_max_gain 0.18

echo === Scheduler comparison (FIFO vs EDF) ===
python -m gqdc.experiments.compare_schedulers --config %CFG% --seeds 5 --fixed_setpoint 6.01 --cool_fraction 1.09 --cop_min_at6 2.32 --cop_max_at12 5.66 --ambient_base_c 21.1 --ambient_swing_c 7.11 --econ_max_gain 0.21

echo === Fairness analysis (with deferrals, protecting priority) ===
python -m gqdc.experiments.fairness_analysis --config %CFG% --p_priority 0.3 --deadline_normal_s 180 --deadline_priority_s 120 --use_deferrals

echo === Stress scenarios ===
python -m gqdc.experiments.stress_scenarios --config %CFG% --seeds 6 --fixed_setpoint 6.01 --cool_fraction 1.09 --cop_min_at6 2.32 --cop_max_at12 5.66 --ambient_base_c 21.1 --ambient_swing_c 7.11 --econ_max_gain 0.21

echo === Cost compare ===
python -m gqdc.experiments.cost_compare --config %CFG% --seeds 5 --fixed_setpoint 6.01 --cool_fraction 1.09 --cop_min_at6 2.32 --cop_max_at12 5.66 --ambient_base_c 21.1 --ambient_swing_c 7.11 --econ_max_gain 0.21

echo === Plots/table refresh ===
python -m gqdc.experiments.plot_figures
python -m gqdc.experiments.make_tables

echo === Build Word (full) ===
python -m gqdc.experiments.build_docx_all

echo Done. Artifacts are in outputs\
pause
