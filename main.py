"""Main entry point for hybrid energy storage planning on IEEE 33-bus system.

Determines optimal locations and capacities for battery (electrochemical) and
hydrogen storage to minimize total annualized cost (investment + operation).

Usage:
    python main.py
    python main.py --n_days 365 --n_clusters 6 --time_limit 3600
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from parse_network import Network
from scenario_generator import (
    generate_profiles,
    cluster_typical_days,
    build_scenario_data,
)
from planning_model import build_and_solve
from results import print_summary, plot_results, export_results


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Storage Planning for IEEE 33-Bus System"
    )
    parser.add_argument("--case_file", default="data/case33bw.m",
                        help="Path to MATPOWER case file")
    parser.add_argument("--n_days", type=int, default=365,
                        help="Number of days for scenario generation")
    parser.add_argument("--n_clusters", type=int, default=6,
                        help="Number of typical day clusters")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Solver time limit (seconds)")
    parser.add_argument("--mip_gap", type=float, default=0.01,
                        help="MIP optimality gap")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    args = parser.parse_args()

    # Step 1: Parse network
    if not args.quiet:
        print("=" * 60)
        print("  HYBRID STORAGE PLANNING FOR IEEE 33-BUS DISTRIBUTION SYSTEM")
        print("=" * 60)
        print("\n[1/4] Loading IEEE 33-bus network...")

    net = Network(args.case_file)
    if not args.quiet:
        net.print_summary()

    # Step 2: Generate scenarios
    if not args.quiet:
        print(f"\n[2/4] Generating {args.n_days} days of load/PV/wind scenarios...")

    profiles = generate_profiles(net, n_days=args.n_days, seed=args.seed)

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

    # Step 3: Build and solve MILP
    if not args.quiet:
        print(f"\n[3/4] Building and solving MILP model "
              f"(time_limit={args.time_limit}s, mip_gap={args.mip_gap})...")

    model, results = build_and_solve(
        net, scenarios,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        verbose=not args.quiet,
    )

    # Step 4: Output results
    if not args.quiet:
        print("\n[4/4] Generating results...")

    print_summary(results, net)
    export_results(results, net, args.output_dir)
    plot_results(results, net, args.output_dir)

    if not args.quiet:
        print("\nDone.")


if __name__ == "__main__":
    main()
