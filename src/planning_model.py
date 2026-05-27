"""Build and solve the hybrid storage planning MILP model with Gurobi."""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# --- Technology cost parameters ---
# Battery (Li-ion) - 2024-2025 utility-scale estimates
C_BAT_E = 150000.0       # $/MWh energy capacity
C_BAT_P = 150000.0       # $/MW power rating
ETA_CH = 0.95            # charging efficiency
ETA_DIS = 0.95           # discharging efficiency
SOC_MIN = 0.10           # min state of charge
SOC_MAX = 0.90           # max state of charge
LT_BAT = 12              # lifetime (years)

# Hydrogen system (PEM) - projected 2025 costs
C_ELZ = 600000.0         # $/MW electrolyzer
C_FC = 800000.0          # $/MW fuel cell
C_TANK = 500.0           # $/kg H2 storage
ETA_ELZ = 0.65           # electrolyzer efficiency (LHV)
ETA_FC = 0.55            # fuel cell efficiency (LHV)
LHV_H2 = 33.33           # kWh/kg H2 (lower heating value)
LT_H2 = 15               # lifetime (years)

# Annualization
DISCOUNT_RATE = 0.06
def crf(r, n):
    """Capital recovery factor: r*(1+r)^n / ((1+r)^n - 1)"""
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)

CRF_BAT = crf(DISCOUNT_RATE, LT_BAT)
CRF_H2 = crf(DISCOUNT_RATE, LT_H2)

# Operational cost
PENALTY_CURTAIL = 0.20   # $/kWh penalty for renewable curtailment

# Big-M values
M_P = 0.5                # p.u. max storage power at a bus (5 MW on 10 MVA base)
M_E = 12.0               # p.u.-h max energy (120 MWh = 24h * 5MW)
M_E_H2_KG = 5000.0       # kg H2 max tank
P_BAT_MIN = 0.001        # p.u. min battery power (10 kW) - must be >0 if installed
P_H2_MIN = 0.005         # p.u. min H2 component power (50 kW)

# Location budget
N_MAX_BAT = 5
N_MAX_H2 = 5

# Voltage limits (squared) — matched to MATPOWER IEEE 33-bus defaults
V_MIN_SQ = 0.90 ** 2
V_MAX_SQ = 1.10 ** 2

# Minimum energy-to-power ratio (hours) - ensures storage is real
MIN_DURATION_BAT = 2.0   # Battery: at least 2-hour duration
MIN_DURATION_H2 = 4.0    # H2: at least 4-hour duration at rated electrolyzer power

# S_base for unit conversions
S_BASE_MW = 10.0


def build_and_solve(net, scenarios, time_limit=3600, mip_gap=0.01, verbose=True):
    """
    Build and solve the hybrid storage planning MILP.

    Args:
        net: Network instance from parse_network
        scenarios: list of scenario dicts from build_scenario_data
        time_limit: solver time limit (seconds)
        mip_gap: MIP optimality gap
        verbose: print solver log

    Returns:
        model: Gurobi model object (after optimization)
        results: dict with extracted solution values
    """
    n_buses = net.n_buses
    n_buses_ns = n_buses - 1  # non-slack buses

    # --- Build model ---
    m = gp.Model("HybridStoragePlanning")
    if not verbose:
        m.Params.OutputFlag = 0
    m.Params.MIPGap = mip_gap
    m.Params.TimeLimit = time_limit
    m.Params.Threads = 8
    m.Params.Method = 2           # Barrier for root
    m.Params.Crossover = 0
    m.Params.Presolve = 2

    # ========== Planning variables ==========
    # Binary: install battery / hydrogen at bus i
    y_bat = m.addVars(n_buses, vtype=GRB.BINARY, name="y_bat")
    y_h2 = m.addVars(n_buses, vtype=GRB.BINARY, name="y_h2")

    # Continuous capacities
    E_bat = m.addVars(n_buses, lb=0, ub=M_E, name="E_bat")        # battery energy (p.u.-h)
    P_bat = m.addVars(n_buses, lb=0, ub=M_P, name="P_bat")        # battery power (p.u.)
    P_elz = m.addVars(n_buses, lb=0, ub=M_P, name="P_elz")        # electrolyzer power (p.u.)
    P_fc = m.addVars(n_buses, lb=0, ub=M_P, name="P_fc")          # fuel cell power (p.u.)
    E_tank = m.addVars(n_buses, lb=0, ub=M_E_H2_KG, name="E_tank") # H2 tank (kg)

    # Big-M coupling for capacities
    for i in range(n_buses):
        m.addConstr(E_bat[i] <= M_E * y_bat[i], f"bigM_Ebat_{i}")
        m.addConstr(P_bat[i] <= M_P * y_bat[i], f"bigM_Pbat_{i}")
        m.addConstr(P_bat[i] >= P_BAT_MIN * y_bat[i], f"minP_bat_{i}")
        m.addConstr(P_elz[i] <= M_P * y_h2[i], f"bigM_Pelz_{i}")
        m.addConstr(P_elz[i] >= P_H2_MIN * y_h2[i], f"minP_elz_{i}")
        m.addConstr(P_fc[i] <= M_P * y_h2[i], f"bigM_Pfc_{i}")
        m.addConstr(P_fc[i] >= P_H2_MIN * y_h2[i], f"minP_fc_{i}")
        m.addConstr(E_tank[i] <= M_E_H2_KG * y_h2[i], f"bigM_Etank_{i}")

    # Location budgets
    m.addConstr(y_bat.sum() <= N_MAX_BAT, "budget_bat")
    m.addConstr(y_h2.sum() <= N_MAX_H2, "budget_h2")

    # Minimum energy-to-power ratio (ensures storage is meaningful)
    K_h2 = S_BASE_MW / LHV_H2 * 1000.0  # kg H2 per p.u.-h
    for i in range(n_buses):
        m.addConstr(E_bat[i] >= MIN_DURATION_BAT * P_bat[i],
                    f"min_dur_bat_{i}")
        m.addConstr(E_tank[i] >= MIN_DURATION_H2 * K_h2 * P_elz[i],
                    f"min_dur_h2_elz_{i}")
        m.addConstr(E_tank[i] >= MIN_DURATION_H2 * K_h2 * P_fc[i],
                    f"min_dur_h2_fc_{i}")

    # No storage at slack bus
    m.addConstr(y_bat[0] == 0, "no_bat_slack")
    m.addConstr(y_h2[0] == 0, "no_h2_slack")

    # ========== Operational variables (per scenario, per hour) ==========
    # We'll index by (i, k, h) where k = scenario, h = hour
    # For efficiency, create variables only when needed
    K = len(scenarios)
    H = scenarios[0]["n_hours"]

    # Store variable dicts keyed by (i, k, h)
    p_ch = {}       # battery charge
    p_dis = {}      # battery discharge
    soc = {}        # battery SOC
    p_elz_op = {}   # electrolyzer operation
    p_fc_op = {}    # fuel cell operation
    m_h2 = {}       # H2 mass in tank
    p_curt = {}     # renewable curtailment

    # Branch power flows keyed by (fbus, tbus, k, h) - only for closed branches
    # We use the downstream formulation, so we directly compute branch flows
    # from nodal injections

    # For each scenario and hour, we need:
    # - Nodal net injection
    # - Branch flows
    # - Voltages

    # Let's create variables for net power injection at each node/time
    P_net = {}      # net active power injection (load - gen + storage) at bus i
    Q_net = {}      # net reactive power injection

    # Branch flows - keyed by branch index
    P_flow_br = {}   # active flow on branch b at (k,h)
    Q_flow_br = {}   # reactive flow on branch b at (k,h)
    P_sub_import = {} # substation import variable (k,h) -> var

    # Squared voltages
    V_sq = {}       # V^2 at bus i for (k,h)

    # Get closed branches as list of (fbus, tbus)
    branches = net.get_closed_branches()
    br_idx_map = {}  # (fbus, tbus) -> branch position index
    for idx, (fbus, tbus, _, _) in enumerate(branches):
        br_idx_map[(fbus, tbus)] = idx

    # Create operational variables for each scenario and hour
    if verbose:
        print(f"Building model with {K} scenarios x {H} hours = {K*H} time periods...")

    for k_scen, scen in enumerate(scenarios):
        for h in range(H):
            key = (k_scen, h)

            # Power flows on each branch
            for (fbus, tbus, _, _) in branches:
                P_flow_br[(fbus, tbus, k_scen, h)] = m.addVar(
                    lb=-GRB.INFINITY, name=f"Pf_{fbus}_{tbus}_{k_scen}_{h}")
                Q_flow_br[(fbus, tbus, k_scen, h)] = m.addVar(
                    lb=-GRB.INFINITY, name=f"Qf_{fbus}_{tbus}_{k_scen}_{h}")

            # Voltages
            for i in range(1, n_buses + 1):
                V_sq[(i, k_scen, h)] = m.addVar(
                    lb=V_MIN_SQ, ub=V_MAX_SQ, name=f"Vsq_{i}_{k_scen}_{h}")

            # Slack voltage = 1.0 p.u.
            m.addConstr(V_sq[(1, k_scen, h)] == 1.0,
                        f"Vslack_{k_scen}_{h}")

            # P_sub_import: auxiliary for one-sided import cost
            # We charge only for imports, not exports
            P_sub_import[(k_scen, h)] = m.addVar(
                lb=0, name=f"Psub_import_{k_scen}_{h}")

            # P_sub_import >= sum of flows from slack (only charge for imports)
            slack_flows = []
            for child in net.children[1]:
                pf = P_flow_br.get((1, child, k_scen, h))
                if pf is not None:
                    slack_flows.append(pf)
            if slack_flows:
                m.addConstr(P_sub_import[(k_scen, h)] >= gp.quicksum(slack_flows),
                            f"Pimport_def_{k_scen}_{h}")

            # Battery operation variables
            for i in range(1, n_buses + 1):
                p_ch[(i, k_scen, h)] = m.addVar(
                    lb=0, ub=M_P, name=f"pch_{i}_{k_scen}_{h}")
                p_dis[(i, k_scen, h)] = m.addVar(
                    lb=0, ub=M_P, name=f"pdis_{i}_{k_scen}_{h}")
                soc[(i, k_scen, h)] = m.addVar(
                    lb=0, ub=M_E, name=f"soc_{i}_{k_scen}_{h}")

                # H2 operation variables
                p_elz_op[(i, k_scen, h)] = m.addVar(
                    lb=0, ub=M_P, name=f"pelz_{i}_{k_scen}_{h}")
                p_fc_op[(i, k_scen, h)] = m.addVar(
                    lb=0, ub=M_P, name=f"pfc_{i}_{k_scen}_{h}")
                m_h2[(i, k_scen, h)] = m.addVar(
                    lb=0, ub=M_E_H2_KG, name=f"mh2_{i}_{k_scen}_{h}")

                # Curtailment
                p_curt[(i, k_scen, h)] = m.addVar(
                    lb=0, name=f"pcurt_{i}_{k_scen}_{h}")

    # ========== Constraints ==========

    # --- Storage operation limits (Big-M coupling to y) ---
    for k_scen in range(K):
        for h in range(H):
            for i in range(1, n_buses + 1):
                m.addConstr(p_ch[(i, k_scen, h)] <= M_P * y_bat[i - 1],
                            f"bigM_ch_{i}_{k_scen}_{h}")
                m.addConstr(p_dis[(i, k_scen, h)] <= M_P * y_bat[i - 1],
                            f"bigM_dis_{i}_{k_scen}_{h}")
                m.addConstr(soc[(i, k_scen, h)] <= M_E * y_bat[i - 1],
                            f"bigM_soc_{i}_{k_scen}_{h}")
                m.addConstr(p_elz_op[(i, k_scen, h)] <= M_P * y_h2[i - 1],
                            f"bigM_elzop_{i}_{k_scen}_{h}")
                m.addConstr(p_fc_op[(i, k_scen, h)] <= M_P * y_h2[i - 1],
                            f"bigM_fcop_{i}_{k_scen}_{h}")
                m.addConstr(m_h2[(i, k_scen, h)] <= M_E_H2_KG * y_h2[i - 1],
                            f"bigM_mh2_{i}_{k_scen}_{h}")

                # Operation limits from rated capacities
                m.addConstr(p_ch[(i, k_scen, h)] <= P_bat[i - 1],
                            f"ch_limit_{i}_{k_scen}_{h}")
                m.addConstr(p_dis[(i, k_scen, h)] <= P_bat[i - 1],
                            f"dis_limit_{i}_{k_scen}_{h}")
                m.addConstr(soc[(i, k_scen, h)] <= SOC_MAX * E_bat[i - 1],
                            f"soc_max_{i}_{k_scen}_{h}")
                m.addConstr(soc[(i, k_scen, h)] >= SOC_MIN * E_bat[i - 1],
                            f"soc_min_{i}_{k_scen}_{h}")
                m.addConstr(p_elz_op[(i, k_scen, h)] <= P_elz[i - 1],
                            f"elz_limit_{i}_{k_scen}_{h}")
                m.addConstr(p_fc_op[(i, k_scen, h)] <= P_fc[i - 1],
                            f"fc_limit_{i}_{k_scen}_{h}")
                m.addConstr(m_h2[(i, k_scen, h)] <= E_tank[i - 1],
                            f"tank_limit_{i}_{k_scen}_{h}")

    # --- Battery SOC dynamics ---
    for k_scen in range(K):
        for h in range(H):
            for i in range(1, n_buses + 1):
                if h == 0:
                    # Initial SOC = 0.5 * capacity
                    m.addConstr(soc[(i, k_scen, 0)] == 0.5 * E_bat[i - 1],
                                f"soc_init_{i}_{k_scen}")
                else:
                    # soc(t) = soc(t-1) + eta_ch * p_ch(t-1) - p_dis(t-1) / eta_dis
                    m.addConstr(
                        soc[(i, k_scen, h)] ==
                        soc[(i, k_scen, h - 1)]
                        + ETA_CH * p_ch[(i, k_scen, h - 1)]
                        - p_dis[(i, k_scen, h - 1)] / ETA_DIS,
                        f"soc_dyn_{i}_{k_scen}_{h}"
                    )

    # Daily cyclic SOC: end of day = start of day
    for k_scen in range(K):
        for i in range(1, n_buses + 1):
            m.addConstr(soc[(i, k_scen, H - 1)]
                        + ETA_CH * p_ch[(i, k_scen, H - 1)]
                        - p_dis[(i, k_scen, H - 1)] / ETA_DIS
                        == soc[(i, k_scen, 0)],
                        f"soc_cyclic_{i}_{k_scen}")

    # --- H2 mass balance dynamics ---
    for k_scen in range(K):
        for h in range(H):
            for i in range(1, n_buses + 1):
                if h == 0:
                    m.addConstr(m_h2[(i, k_scen, 0)] == 0.5 * E_tank[i - 1],
                                f"h2_init_{i}_{k_scen}")
                else:
                    # m(t) = m(t-1) + K_h2 * (eta_elz * p_elz - p_fc / eta_fc)
                    m.addConstr(
                        m_h2[(i, k_scen, h)] ==
                        m_h2[(i, k_scen, h - 1)]
                        + K_h2 * (ETA_ELZ * p_elz_op[(i, k_scen, h - 1)]
                                  - p_fc_op[(i, k_scen, h - 1)] / ETA_FC),
                        f"h2_dyn_{i}_{k_scen}_{h}"
                    )

    # Daily cyclic H2
    for k_scen in range(K):
        for i in range(1, n_buses + 1):
            m.addConstr(m_h2[(i, k_scen, H - 1)]
                        + K_h2 * (ETA_ELZ * p_elz_op[(i, k_scen, H - 1)]
                                  - p_fc_op[(i, k_scen, H - 1)] / ETA_FC)
                        == m_h2[(i, k_scen, 0)],
                        f"h2_cyclic_{i}_{k_scen}")

    # --- Nodal power balance ---
    # For each bus, at each time:
    # Net injection = storage_discharge - storage_charge
    #   + fuel_cell - electrolyzer - curtailment
    #   + pv + wind - load
    #
    #   P_net[i] = p_dis[i] - p_ch[i] + p_fc[i] - p_elz[i] - p_curt[i]
    #              + pv_pu[i] + wind_pu[i] - load_pu[i]

    for k_scen, scen in enumerate(scenarios):
        load_pu = scen["load_pu"]
        q_load_pu = scen["q_load_pu"]
        pv_pu = scen["pv_pu"]
        wind_pu = scen["wind_pu"]

        for h in range(H):
            for i in range(1, n_buses + 1):
                bus_idx = i - 1
                P_inj = (load_pu[bus_idx, h]
                         - pv_pu[bus_idx, h] - wind_pu[bus_idx, h]
                         - p_dis[(i, k_scen, h)] + p_ch[(i, k_scen, h)]
                         - p_fc_op[(i, k_scen, h)] + p_elz_op[(i, k_scen, h)]
                         + p_curt[(i, k_scen, h)])

                Q_inj = q_load_pu[bus_idx, h]

                # P_net as expression for branch flow constraint
                # We don't need separate P_net variables; use them directly in constraints

                # Branch flow constraint:
                # Sum of flows into bus = net injection at bus
                # For each bus i (i != 1):
                #   P_flow(parent->i) - sum_{child} P_flow(i->child) = P_inj

                # Find parent and children
                if i == 1:
                    # Slack bus: net injection is substation import
                    pass
                else:
                    parent = net.parent[i]
                    # Flow from parent to i
                    p_in = P_flow_br.get((parent, i, k_scen, h))
                    if p_in is None:
                        p_in = P_flow_br.get((i, parent, k_scen, h))
                        # Flow convention: positive = from fbus to tbus
                        # If (i, parent) exists, flow is negative
                        if p_in is not None:
                            p_in = -p_in  # reverse direction

                    # Sum of flows to children
                    p_out = 0.0
                    for child in net.children[i]:
                        pf = P_flow_br.get((i, child, k_scen, h))
                        if pf is not None:
                            p_out += pf

                    if p_in is not None:
                        m.addConstr(p_in - p_out == P_inj,
                                    f"Pbal_{i}_{k_scen}_{h}")

                # Reactive power balance
                if i != 1:
                    parent = net.parent[i]
                    q_in = Q_flow_br.get((parent, i, k_scen, h))
                    if q_in is None:
                        q_in = Q_flow_br.get((i, parent, k_scen, h))
                        if q_in is not None:
                            q_in = -q_in

                    q_out = 0.0
                    for child in net.children[i]:
                        qf = Q_flow_br.get((i, child, k_scen, h))
                        if qf is not None:
                            q_out += qf

                    if q_in is not None:
                        m.addConstr(q_in - q_out == Q_inj,
                                    f"Qbal_{i}_{k_scen}_{h}")

    # --- LinDistFlow voltage drop ---
    # For each branch (i->j): V_sq[j] = V_sq[i] - 2*(r*P + x*Q)
    for k_scen in range(K):
        for h in range(H):
            for (fbus, tbus, r, x) in branches:
                pf = P_flow_br[(fbus, tbus, k_scen, h)]
                qf = Q_flow_br[(fbus, tbus, k_scen, h)]
                m.addConstr(
                    V_sq[(tbus, k_scen, h)] ==
                    V_sq[(fbus, k_scen, h)]
                    - 2 * (r * pf + x * qf),
                    f"Vdrop_{fbus}_{tbus}_{k_scen}_{h}"
                )

    # --- Curtailment limit ---
    for k_scen, scen in enumerate(scenarios):
        pv_pu = scen["pv_pu"]
        wind_pu = scen["wind_pu"]
        for h in range(H):
            for i in range(1, n_buses + 1):
                bus_idx = i - 1
                max_re = pv_pu[bus_idx, h] + wind_pu[bus_idx, h]
                m.addConstr(p_curt[(i, k_scen, h)] <= max_re,
                            f"curt_lim_{i}_{k_scen}_{h}")

    # --- No storage at buses with no generation/load (optional) ---
    # Not strictly necessary

    # ========== Objective ==========
    # Investment cost (annualized)
    obj_inv = gp.QuadExpr()

    for i in range(1, n_buses):
        bus_idx = i
        # Battery investment
        obj_inv += CRF_BAT * (C_BAT_E * E_bat[bus_idx] * S_BASE_MW
                               + C_BAT_P * P_bat[bus_idx] * S_BASE_MW)
        # H2 investment
        obj_inv += CRF_H2 * (C_ELZ * P_elz[bus_idx] * S_BASE_MW
                              + C_FC * P_fc[bus_idx] * S_BASE_MW
                              + C_TANK * E_tank[bus_idx])

    # Operational cost
    obj_op = gp.LinExpr()
    days_per_year = 365

    for k_scen, scen in enumerate(scenarios):
        w = scen["weight"]
        price = scen["price_per_kwh"]

        for h in range(H):
            # Grid import cost: P_sub_import(t) * price(t) [one-sided, no export revenue]
            p_import = P_sub_import.get((k_scen, h))
            if p_import is not None:
                obj_op += (days_per_year * w * p_import * S_BASE_MW
                           * price[h] * 1000.0)

            # Curtailment penalty
            for i in range(1, n_buses + 1):
                obj_op += (days_per_year * w
                           * p_curt[(i, k_scen, h)] * S_BASE_MW
                           * PENALTY_CURTAIL * 1000.0)

    m.setObjective(obj_inv + obj_op, GRB.MINIMIZE)

    if verbose:
        print(f"Model built: {m.NumVars} variables, {m.NumConstrs} constraints")
        print("Solving...")

    m.optimize()

    # Handle infeasibility gracefully
    if m.Status == GRB.INFEASIBLE:
        if verbose:
            print("Model is INFEASIBLE — computing IIS for diagnostics...")
            m.computeIIS()
            iis_file = "results/model_iis.ilp"
            m.write(iis_file)
            print(f"  IIS written to {iis_file}")
        return m, {
            "status": m.Status,
            "obj_val": None,
            "mip_gap": None,
            "solve_time": m.Runtime,
            "battery": {},
            "hydrogen": {},
            "cost_breakdown": {},
            "operation": {},
        }

    # ========== Extract results ==========
    results = _extract_results(m, net, y_bat, y_h2, E_bat, P_bat, P_elz, P_fc,
                               E_tank, scenarios, soc, p_ch, p_dis, m_h2,
                               p_elz_op, p_fc_op, p_curt, V_sq, K, H, branches,
                               P_flow_br, Q_flow_br)

    return m, results


def _extract_results(m, net, y_bat, y_h2, E_bat, P_bat, P_elz, P_fc, E_tank,
                     scenarios, soc, p_ch, p_dis, m_h2_stor, p_elz_op, p_fc_op,
                     p_curt, V_sq, K, H, branches, P_flow_br, Q_flow_br):
    """Extract solution from optimized model."""

    results = {
        "status": m.Status,
        "obj_val": m.ObjVal if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL
                   else None,
        "mip_gap": m.MIPGap,
        "solve_time": m.Runtime,
        "battery": {},
        "hydrogen": {},
        "cost_breakdown": {},
        "operation": {},
    }

    S = S_BASE_MW

    # Battery placements
    for i in range(net.n_buses):
        if y_bat[i].X > 0.5:
            bus_id = i + 1
            results["battery"][bus_id] = {
                "P_MW": round(P_bat[i].X * S, 4),
                "E_MWh": round(E_bat[i].X * S, 4),
                "duration_h": round(E_bat[i].X / max(P_bat[i].X, 1e-6), 2),
            }

    # Hydrogen placements
    for i in range(net.n_buses):
        if y_h2[i].X > 0.5:
            bus_id = i + 1
            results["hydrogen"][bus_id] = {
                "P_elz_MW": round(P_elz[i].X * S, 4),
                "P_fc_MW": round(P_fc[i].X * S, 4),
                "E_tank_kg": round(E_tank[i].X, 4),
                "E_tank_MWh": round(E_tank[i].X * LHV_H2 / 1000.0, 4),
            }

    # Cost breakdown
    inv_bat = 0.0
    inv_h2 = 0.0
    for i in range(net.n_buses):
        inv_bat += CRF_BAT * (C_BAT_E * E_bat[i].X * S + C_BAT_P * P_bat[i].X * S)
        inv_h2 += CRF_H2 * (C_ELZ * P_elz[i].X * S + C_FC * P_fc[i].X * S
                            + C_TANK * E_tank[i].X)

    results["cost_breakdown"] = {
        "inv_battery_annual": round(inv_bat, 2),
        "inv_hydrogen_annual": round(inv_h2, 2),
        "inv_total_annual": round(inv_bat + inv_h2, 2),
        "obj_total": round(m.ObjVal, 2) if m.ObjVal is not None else None,
    }

    # Operational data for the first typical day (for visualization)
    if K > 0 and m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        k = 0
        op = {
            "k_scen": k,
            "weight": scenarios[k]["weight"],
            "hours": list(range(H)),
            "load_total_MW": [],
            "pv_total_MW": [],
            "wind_total_MW": [],
            "bat_charge_total_MW": [],
            "bat_discharge_total_MW": [],
            "h2_elz_total_MW": [],
            "h2_fc_total_MW": [],
            "v_min_profile": [],
            "v_max_profile": [],
            "price": scenarios[k]["price_per_kwh"],
        }
        for h in range(H):
            ld = sum(scenarios[k]["load_pu"][bus, h] for bus in range(net.n_buses)) * S
            pv = sum(scenarios[k]["pv_pu"][bus, h] for bus in range(net.n_buses)) * S
            wd = sum(scenarios[k]["wind_pu"][bus, h] for bus in range(net.n_buses)) * S
            ch = sum(p_ch[(i + 1, k, h)].X for i in range(net.n_buses)) * S
            disch = sum(p_dis[(i + 1, k, h)].X for i in range(net.n_buses)) * S
            elz_t = sum(p_elz_op[(i + 1, k, h)].X for i in range(net.n_buses)) * S
            fc_t = sum(p_fc_op[(i + 1, k, h)].X for i in range(net.n_buses)) * S
            v_vals = [np.sqrt(V_sq[(i + 1, k, h)].X) for i in range(net.n_buses)]

            op["load_total_MW"].append(ld)
            op["pv_total_MW"].append(pv)
            op["wind_total_MW"].append(wd)
            op["bat_charge_total_MW"].append(ch)
            op["bat_discharge_total_MW"].append(disch)
            op["h2_elz_total_MW"].append(elz_t)
            op["h2_fc_total_MW"].append(fc_t)
            op["v_min_profile"].append(min(v_vals))
            op["v_max_profile"].append(max(v_vals))

        results["operation"] = op

    return results


if __name__ == "__main__":
    from parse_network import Network
    from scenario_generator import generate_profiles, cluster_typical_days, build_scenario_data

    print("Loading network...")
    net = Network("data/case33bw.m")
    net.print_summary()

    print("\nGenerating scenarios...")
    profiles = generate_profiles(net, n_days=90, seed=42)
    typical = cluster_typical_days(profiles, n_clusters=4)
    scenarios = build_scenario_data(profiles, typical, net)
    print(f"  Clustered into {len(scenarios)} typical days")
    for i, s in enumerate(scenarios):
        print(f"    Day {i+1}: weight={s['weight']:.3f}")

    print("\nBuilding and solving MILP...")
    model, results = build_and_solve(net, scenarios, time_limit=600, mip_gap=0.02)

    print("\n" + "=" * 60)
    print("OPTIMAL HYBRID STORAGE PLANNING RESULTS")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Objective: ${results['obj_val']:,.2f}" if results['obj_val'] else "N/A")
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
              f"Tank={data['E_tank_kg']:.1f} kg ({data['E_tank_MWh']:.2f} MWh)")

    print("\n--- Cost Breakdown ---")
    for k, v in results["cost_breakdown"].items():
        print(f"  {k}: ${v:,.2f}")
