"""Main entry point for hybrid energy storage planning on IEEE 33-bus system.

Supports two modes:
  - Deterministic: K-means cluster typical days, single MILP
  - DRO (Wasserstein): N representative samples with adversarial perturbations

Usage:
    python main.py                                      # deterministic, 6 clusters
    python main.py --dro                                # DRO, 20 samples
    python main.py --dro --n_samples 10 --beta 0.05     # DRO with custom params
    python main.py --dro --epsilon 0.15                 # DRO with manual epsilon
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from parse_network import Network
from scenario_generator import (
    generate_profiles,
    cluster_typical_days,
    build_scenario_data,
    build_dro_scenario_data,
)
from planning_model import build_and_solve
from dro_model import build_and_solve_dro
from results import print_summary, plot_results, export_results


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Storage Planning for IEEE 33-Bus System"
    )

    # Network and data
    parser.add_argument("--case_file", default="data/case33bw.m",
                        help="Path to MATPOWER case file")
    parser.add_argument("--n_days", type=int, default=365,
                        help="Number of days for scenario generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Deterministic mode
    parser.add_argument("--n_clusters", type=int, default=6,
                        help="[Deterministic] Number of typical day clusters")

    # DRO mode
    parser.add_argument("--dro", action="store_true",
                        help="Enable Wasserstein DRO mode")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="[DRO] Number of representative samples")
    parser.add_argument("--beta", type=float, default=0.10,
                        help="[DRO] Confidence level for epsilon (default 0.10)")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="[DRO] Manual Wasserstein radius (MW). "
                             "Auto-computed if not specified.")

    # Solver
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Solver time limit (seconds)")
    parser.add_argument("--mip_gap", type=float, default=0.01,
                        help="MIP optimality gap")

    # Output
    parser.add_argument("--output_dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    args = parser.parse_args()

    # --- Step 1: Parse network ---
    if not args.quiet:
        mode = "DRO (Wasserstein)" if args.dro else "Deterministic"
        print("=" * 60)
        print(f"  HYBRID STORAGE PLANNING — {mode}")
        print("=" * 60)
        print("\n[1/4] Loading IEEE 33-bus network...")

    net = Network(args.case_file)
    if not args.quiet:
        net.print_summary()

    # --- Step 2: Generate scenarios ---
    if not args.quiet:
        print(f"\n[2/4] Generating {args.n_days} days of load/PV/wind profiles...")

    profiles = generate_profiles(net, n_days=args.n_days, seed=args.seed)

    if args.dro:
        if not args.quiet:
            print(f"  Building DRO dataset with N={args.n_samples} samples...")

        dro_data = build_dro_scenario_data(
            net, profiles, N=args.n_samples, beta=args.beta, seed=args.seed)

        # Override epsilon if manually specified
        if args.epsilon is not None:
            dro_data["epsilon"] = args.epsilon

        if not args.quiet:
            print(f"  Selected {dro_data['N']} representative days")
            print(f"  Wasserstein radius ε = {dro_data['epsilon']:.4f} MW")
            print(f"  β = {dro_data['beta']}, "
                  f"Formula: ε = σ_ref · (log(1/β)/N)^(1/d)")

        # --- Step 3: Build and solve DRO MILP ---
        if not args.quiet:
            print(f"\n[3/4] Building and solving DRO MILP "
                  f"(time_limit={args.time_limit}s, mip_gap={args.mip_gap})...")

        model, results = build_and_solve_dro(
            net, dro_data,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            verbose=not args.quiet,
        )

    else:
        if not args.quiet:
            print(f"  Clustering into {args.n_clusters} typical days...")

        typical = cluster_typical_days(profiles, n_clusters=args.n_clusters,
                                       seed=args.seed)
        scenarios = build_scenario_data(profiles, typical, net)

        if not args.quiet:
            print(f"  Created {len(scenarios)} typical day scenarios:")
            for i, s in enumerate(scenarios):
                print(f"    Day type {i+1}: weight = {s['weight']:.3f} "
                      f"({s['weight']*args.n_days:.0f} days/year)")

        # --- Step 3: Build and solve deterministic MILP ---
        if not args.quiet:
            print(f"\n[3/4] Building and solving MILP "
                  f"(time_limit={args.time_limit}s, mip_gap={args.mip_gap})...")

        model, results = build_and_solve(
            net, scenarios,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            verbose=not args.quiet,
        )

    # --- Step 4: Output results ---
    if not args.quiet:
        print("\n[4/4] Generating results...")

    print_summary(results, net)
    export_results(results, net, args.output_dir)
    plot_results(results, net, args.output_dir)

    if not args.quiet:
        print("\nDone.")


if __name__ == "__main__":
    main()
