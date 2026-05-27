"""Wasserstein DRO storage planning — thin wrapper around planning_model.

The ∞-norm Wasserstein DRO worst-case perturbations are pre-computed in
scenario_generator.build_dro_scenario_data():
  - load_wc  = load_nominal + epsilon   (more load = higher cost)
  - pv_wc    = max(0, pv_nominal - epsilon)
  - wind_wc  = max(0, wind_nominal - epsilon)

These are allocated to buses via fixed ratios and passed as scenarios to
the standard deterministic planning model. This is exact for ∞-norm
Wasserstein DRO with monotonic cost structure.
"""

import numpy as np
from planning_model import build_and_solve


def build_and_solve_dro(net, dro_data, time_limit=3600, mip_gap=0.01,
                         verbose=True):
    """Build and solve the Wasserstein DRO storage planning MILP.

    Args:
        net: Network instance from parse_network
        dro_data: dict from build_dro_scenario_data with keys:
            scenarios (list of worst-case-adjusted scenario dicts),
            epsilon, N, beta, sample_indices
        time_limit: solver time limit (seconds)
        mip_gap: MIP optimality gap
        verbose: print solver log

    Returns:
        model: Gurobi model object
        results: dict with extracted solution values + dro_info
    """
    scenarios = dro_data["scenarios"]
    epsilon = dro_data["epsilon"]

    if verbose:
        print(f"DRO model: N={len(scenarios)} samples, "
              f"epsilon={epsilon:.4f} MW, beta={dro_data.get('beta', 0.10)}")
        print(f"  Worst-case: load_nom+eps, pv_nom-eps, wind_nom-eps (clamped >=0)")

    # Use the standard deterministic model with worst-case-adjusted data
    model, results = build_and_solve(
        net, scenarios,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )

    # Add DRO metadata
    results["dro_info"] = {
        "epsilon_MW": epsilon,
        "n_samples": len(scenarios),
        "beta": dro_data.get("beta", None),
        "sample_indices": [int(x) for x in dro_data.get("sample_indices", [])],
    }

    # Add worst-case vs nominal comparison for sample 0
    k0 = 0
    if len(scenarios) > 0 and "load_nominal_total" in scenarios[k0]:
        s0 = scenarios[0]
        results["dro_info"]["worst_case_sample_0"] = {
            "load_nominal": s0["load_nominal_total"].tolist(),
            "load_worstcase": s0["load_wc_total"].tolist(),
            "pv_nominal": s0["pv_nominal_total"].tolist(),
            "pv_worstcase": s0["pv_wc_total"].tolist(),
            "wind_nominal": s0["wind_nominal_total"].tolist(),
            "wind_worstcase": s0["wind_wc_total"].tolist(),
        }

    return model, results


if __name__ == "__main__":
    from parse_network import Network
    from scenario_generator import generate_profiles, build_dro_scenario_data

    print("Loading network...")
    net = Network("data/case33bw.m")
    net.print_summary()

    print("\nGenerating DRO scenarios...")
    profiles = generate_profiles(net, n_days=365, seed=42)
    dro_data = build_dro_scenario_data(net, profiles, N=20, beta=0.10, seed=42)

    print(f"  Selected N={dro_data['N']} samples")
    print(f"  Wasserstein radius epsilon = {dro_data['epsilon']:.4f} MW")
    print(f"  Beta = {dro_data['beta']}")

    print("\nBuilding and solving DRO MILP...")
    model, results = build_and_solve_dro(net, dro_data, time_limit=600,
                                          mip_gap=0.02)

    print("\n" + "=" * 60)
    print("DRO HYBRID STORAGE PLANNING RESULTS")
    print("=" * 60)
    status_map = {2: "OPTIMAL", 13: "SUBOPTIMAL"}
    print(f"Status: {status_map.get(results['status'], results['status'])}")
    if results['obj_val']:
        print(f"Objective: ${results['obj_val']:,.2f}")
    print(f"MIP Gap: {results['mip_gap']:.4f}")
    print(f"Solve time: {results['solve_time']:.1f}s")

    print("\n--- Battery Locations ---")
    for bus, data in results["battery"].items():
        print(f"  Bus {bus}: {data['P_MW']:.3f} MW / {data['E_MWh']:.3f} MWh "
              f"({data['duration_h']:.1f}h)")

    print("\n--- Hydrogen Locations ---")
    for bus, data in results["hydrogen"].items():
        print(f"  Bus {bus}: Elz={data['P_elz_MW']:.3f} MW, "
              f"FC={data['P_fc_MW']:.3f} MW, "
              f"Tank={data['E_tank_kg']:.1f} kg")

    print("\n--- Cost Breakdown ---")
    for k, v in results["cost_breakdown"].items():
        print(f"  {k}: ${v:,.2f}")

    di = results.get("dro_info", {})
    if di:
        print(f"\n--- DRO Info ---")
        print(f"  ε = {di['epsilon_MW']:.4f} MW, N = {di['n_samples']}")
