"""Extract, format, and visualize planning results."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_summary(results, net):
    """Print formatted summary of optimization results to console."""
    print("\n" + "=" * 65)
    print("  HYBRID ENERGY STORAGE PLANNING RESULTS")
    print("=" * 65)

    status_map = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
                  5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT",
                  8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT",
                  11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL"}
    status_str = status_map.get(results["status"], f"CODE_{results['status']}")
    print(f"  Solve status:     {status_str}")
    if results.get("obj_val") is not None:
        print(f"  Total annual cost: ${results['obj_val']:,.2f}")
    if results.get("mip_gap") is not None:
        print(f"  MIP gap:           {results['mip_gap']:.4%}")
    print(f"  Solve time:        {results['solve_time']:.1f} s")

    # Battery
    print("\n  " + "-" * 55)
    print("  BATTERY STORAGE LOCATIONS")
    print(f"  {'Bus':<6} {'Power (MW)':<14} {'Energy (MWh)':<15} {'Duration (h)':<14}")
    print("  " + "-" * 55)
    total_bat_p = 0.0
    total_bat_e = 0.0
    if results["battery"]:
        for bus in sorted(results["battery"].keys()):
            d = results["battery"][bus]
            print(f"  {bus:<6} {d['P_MW']:<14.4f} {d['E_MWh']:<15.4f} {d['duration_h']:<14.1f}")
            total_bat_p += d["P_MW"]
            total_bat_e += d["E_MWh"]
        print(f"  {'Total':<6} {total_bat_p:<14.4f} {total_bat_e:<15.4f}")
    else:
        print("  (No battery installed)")

    # Hydrogen
    print("\n  " + "-" * 65)
    print("  HYDROGEN STORAGE LOCATIONS")
    print(f"  {'Bus':<6} {'Elz (MW)':<12} {'FC (MW)':<12} {'Tank (kg)':<14} {'Tank (MWh)':<14}")
    print("  " + "-" * 65)
    total_elz = 0.0
    total_fc = 0.0
    total_tank = 0.0
    if results["hydrogen"]:
        for bus in sorted(results["hydrogen"].keys()):
            d = results["hydrogen"][bus]
            print(f"  {bus:<6} {d['P_elz_MW']:<12.4f} {d['P_fc_MW']:<12.4f} "
                  f"{d['E_tank_kg']:<14.1f} {d['E_tank_MWh']:<14.4f}")
            total_elz += d["P_elz_MW"]
            total_fc += d["P_fc_MW"]
            total_tank += d["E_tank_kg"]
        print(f"  {'Total':<6} {total_elz:<12.4f} {total_fc:<12.4f} "
              f"{total_tank:<14.1f}")
    else:
        print("  (No hydrogen system installed)")

    # Cost breakdown
    print("\n  " + "-" * 45)
    print("  COST BREAKDOWN (Annualized)")
    print("  " + "-" * 45)
    cb = results["cost_breakdown"]
    for label, key in [("Battery investment", "inv_battery_annual"),
                       ("Hydrogen investment", "inv_hydrogen_annual"),
                       ("Total investment", "inv_total_annual")]:
        if key in cb and cb[key] is not None:
            print(f"  {label:<25s}: ${cb[key]:>15,.2f}")
    if cb.get("obj_total") is not None:
        print(f"  {'Total (incl. operation)':<25s}: ${cb['obj_total']:>15,.2f}")
    print("=" * 65)


def plot_results(results, net, output_dir="results"):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Storage capacity bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Battery
    ax = axes[0]
    buses = list(range(2, net.n_buses + 1))
    bat_p = [results["battery"].get(b, {}).get("P_MW", 0) for b in buses]
    bat_e = [results["battery"].get(b, {}).get("E_MWh", 0) for b in buses]
    x = np.arange(len(buses))
    width = 0.35
    ax.bar(x - width/2, bat_p, width, color="#2196F3", label="Power (MW)")
    ax.bar(x + width/2, bat_e, width, color="#90CAF9", label="Energy (MWh)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in buses], fontsize=8)
    ax.set_title("Battery Storage Capacity per Bus")
    ax.set_xlabel("Bus")
    ax.set_ylabel("Capacity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Hydrogen
    ax = axes[1]
    h2_buses = [results["hydrogen"].get(b, {}).get("P_elz_MW", 0) for b in buses]
    h2_fc = [results["hydrogen"].get(b, {}).get("P_fc_MW", 0) for b in buses]
    ax.bar(x - width/2, h2_buses, width, color="#4CAF50", label="Electrolyzer (MW)")
    ax.bar(x + width/2, h2_fc, width, color="#A5D6A7", label="Fuel Cell (MW)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in buses], fontsize=8)
    ax.set_title("Hydrogen Storage Capacity per Bus")
    ax.set_xlabel("Bus")
    ax.set_ylabel("Capacity (MW)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "storage_capacity.png"), dpi=150)
    plt.close()

    # 2. Operational time series (first typical day)
    op = results.get("operation", {})
    if op:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        h = op["hours"]

        # Top: load and renewable generation
        ax = axes[0]
        ax.fill_between(h, 0, op["load_total_MW"], alpha=0.3, color="blue",
                         label="Load")
        ax.plot(h, op["pv_total_MW"], "r-", linewidth=1.5, label="PV")
        ax.plot(h, op["wind_total_MW"], "g-", linewidth=1.5, label="Wind")
        net_load = [op["load_total_MW"][t] - op["pv_total_MW"][t]
                    - op["wind_total_MW"][t] for t in h]
        ax.plot(h, net_load, "k--", linewidth=1, label="Net load", alpha=0.7)
        ax.set_ylabel("Power (MW)")
        ax.set_title("Load and Renewable Generation (Typical Day)")
        ax.legend(fontsize=8, ncol=4)
        ax.grid(alpha=0.3)

        # Middle: storage operation
        ax = axes[1]
        ax.fill_between(h, 0, op["bat_charge_total_MW"], alpha=0.4,
                         color="#2196F3", label="Battery charge")
        ax.fill_between(h, 0, [-x for x in op["bat_discharge_total_MW"]],
                         alpha=0.4, color="#FF9800", label="Battery discharge")
        ax.fill_between(h, 0, op["h2_elz_total_MW"], alpha=0.4,
                         color="#4CAF50", label="H2 electrolyzer")
        ax.fill_between(h, 0, [-x for x in op["h2_fc_total_MW"]],
                         alpha=0.4, color="#F44336", label="H2 fuel cell")
        ax.set_ylabel("Power (MW)")
        ax.set_title("Storage Operation")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Bottom: voltage profile
        ax = axes[2]
        ax.fill_between(h, op["v_min_profile"], op["v_max_profile"],
                         alpha=0.3, color="green", label="Voltage range")
        ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(y=0.95, color="red", linewidth=0.5, linestyle=":",
                    label="Vmin (0.95)")
        ax.axhline(y=1.05, color="red", linewidth=0.5, linestyle=":",
                    label="Vmax (1.05)")
        ax.set_ylabel("Voltage (p.u.)")
        ax.set_xlabel("Hour")
        ax.set_title("Voltage Profile")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "operation_profile.png"), dpi=150)
        plt.close()

    # 3. Cost breakdown pie chart
    fig, ax = plt.subplots(figsize=(7, 5))
    cb = results["cost_breakdown"]
    labels = ["Battery Invest.", "Hydrogen Invest."]
    values = [cb.get("inv_battery_annual", 0), cb.get("inv_hydrogen_annual", 0)]
    # Remove zero entries
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    if filtered:
        labels, values = zip(*filtered)
        colors = ["#2196F3", "#4CAF50"]
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)],
               startangle=90, explode=[0.02] * len(labels))
        ax.set_title("Annualized Investment Cost Breakdown")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cost_breakdown.png"), dpi=150)
    plt.close()

    # 4. Network topology with storage locations
    fig, ax = plt.subplots(figsize=(16, 5))
    _plot_network_map(ax, net, results)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "network_map.png"), dpi=150)
    plt.close()


def _plot_network_map(ax, net, results):
    """Schematic network topology with storage locations."""
    # Create bus coordinates using a simple layered layout
    pos = {}
    # Manual layout for IEEE 33 (rough schematic for visualization)
    # Main feeder
    main_path = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # Branches off the main path
    branch_1 = [19, 20, 21, 22]
    branch_2 = [23, 24, 25]
    branch_3 = [26, 27, 28, 29, 30, 31, 32, 33]

    # Layout: x along main path, y varies
    for idx, bus in enumerate(main_path):
        pos[bus] = (idx, 0)
    for idx, bus in enumerate(branch_1):
        pos[bus] = (idx + 1, 1.5)
    for idx, bus in enumerate(branch_2):
        pos[bus] = (idx + 3, -1.5)
    for idx, bus in enumerate(branch_3):
        pos[bus] = (idx + 6, -2.5)

    # Draw branches
    for (fbus, tbus, _, _) in net.get_closed_branches():
        if fbus in pos and tbus in pos:
            x1, y1 = pos[fbus]
            x2, y2 = pos[tbus]
            ax.plot([x1, x2], [y1, y2], "gray", linewidth=0.8, alpha=0.6)

    # Draw buses
    bat_buses = set(results["battery"].keys())
    h2_buses = set(results["hydrogen"].keys())
    both_buses = bat_buses & h2_buses

    for bus, (x, y) in pos.items():
        if bus in both_buses:
            color = "#9C27B0"
            size = 120
        elif bus in bat_buses:
            color = "#2196F3"
            size = 90
        elif bus in h2_buses:
            color = "#4CAF50"
            size = 90
        elif bus == 1:
            color = "#FF5722"
            size = 80
        else:
            color = "#BDBDBD"
            size = 40
        ax.scatter(x, y, c=color, s=size, zorder=5, edgecolors="black",
                   linewidth=0.5)
        offset = 0.2 if y >= 0 else -0.3
        ax.annotate(str(bus), (x, y + offset), fontsize=7, ha="center")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722",
               markersize=8, label="Slack bus"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="Battery"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="Hydrogen"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9C27B0",
               markersize=8, label="Both"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.set_title("IEEE 33-Bus Network with Optimal Storage Placement")
    ax.axis("equal")
    ax.axis("off")


def export_results(results, net, output_dir="results"):
    """Export results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON export
    json_data = {
        "status": results["status"],
        "objective": results["obj_val"],
        "mip_gap": results["mip_gap"],
        "solve_time_s": results["solve_time"],
        "battery": results["battery"],
        "hydrogen": results["hydrogen"],
        "cost_breakdown": results["cost_breakdown"],
    }
    if "dro_info" in results:
        json_data["dro_info"] = results["dro_info"]
    with open(os.path.join(output_dir, "placement.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # CSV export
    with open(os.path.join(output_dir, "summary.csv"), "w") as f:
        f.write("Type,Bus,Param,Value\n")
        for bus, d in results["battery"].items():
            f.write(f"Battery,{bus},P_MW,{d['P_MW']}\n")
            f.write(f"Battery,{bus},E_MWh,{d['E_MWh']}\n")
            f.write(f"Battery,{bus},Duration_h,{d['duration_h']}\n")
        for bus, d in results["hydrogen"].items():
            f.write(f"Hydrogen,{bus},P_elz_MW,{d['P_elz_MW']}\n")
            f.write(f"Hydrogen,{bus},P_fc_MW,{d['P_fc_MW']}\n")
            f.write(f"Hydrogen,{bus},E_tank_kg,{d['E_tank_kg']}\n")
            f.write(f"Hydrogen,{bus},E_tank_MWh,{d['E_tank_MWh']}\n")

    print(f"\nResults exported to {output_dir}/")
