"""Parse IEEE 33-bus data from MATPOWER .m file and build radial topology."""

import re
import numpy as np
from collections import deque

# System base values
S_BASE = 10.0       # MVA
V_BASE = 12.66      # kV
Z_BASE = V_BASE ** 2 / S_BASE   # ohms


class Network:
    """Container for IEEE 33-bus network data."""

    def __init__(self, case_file="data/case33bw.m"):
        self.n_buses = 0
        self.n_branches = 0
        self.S_base = S_BASE
        self.V_base = V_BASE
        self.Z_base = Z_BASE
        self.bus_data = {}          # bus_id -> {type, Pd, Qd, ...}
        self.branch_list = []       # list of (fbus, tbus, r, x, status)
        self.parent = {}            # bus -> parent bus (for radial topology)
        self.children = {}          # bus -> list of child buses
        self.downstream = {}        # bus -> set of buses in its subtree
        self.branch_idx = {}        # (fbus, tbus) -> branch index
        self.r_pu = {}              # (fbus, tbus) -> resistance in p.u.
        self.x_pu = {}              # (fbus, tbus) -> reactance in p.u.

        self._parse(case_file)
        self._build_topology()
        self._compute_downstream_sets()

    def _parse(self, filepath):
        """Extract bus and branch data from MATPOWER .m file."""
        with open(filepath, "r") as f:
            content = f.read()

        # Parse bus matrix
        bus_match = re.search(r"mpc\.bus\s*=\s*\[(.*?)\];", content, re.DOTALL)
        if bus_match:
            bus_text = bus_match.group(1)
            rows = self._parse_matrix_rows(bus_text)
            for row in rows:
                bus_i = int(row[0])
                bus_type = int(row[1])
                Pd = float(row[2]) / 1e3  # Convert kW -> MW
                Qd = float(row[3]) / 1e3  # Convert kVAr -> MVAr
                self.bus_data[bus_i] = {
                    "type": bus_type,  # 1=PQ, 2=PV, 3=ref
                    "Pd": Pd,           # MW
                    "Qd": Qd,           # MVAr
                    "Vm": float(row[7]),
                    "Va": float(row[8]),
                    "baseKV": float(row[9]),
                    "Vmax": float(row[11]),
                    "Vmin": float(row[12]),
                }
                self.n_buses = max(self.n_buses, bus_i)

        # Parse branch matrix
        br_match = re.search(r"mpc\.branch\s*=\s*\[(.*?)\];", content, re.DOTALL)
        if br_match:
            br_text = br_match.group(1)
            rows = self._parse_matrix_rows(br_text)
            idx = 0
            for row in rows:
                fbus = int(row[0])
                tbus = int(row[1])
                r_ohm = float(row[2])
                x_ohm = float(row[3])
                r = r_ohm / self.Z_base  # Convert ohms -> p.u.
                x = x_ohm / self.Z_base
                status = int(row[10]) if len(row) > 10 else 1
                self.branch_list.append((fbus, tbus, r, x, status))
                self.branch_idx[(fbus, tbus)] = idx
                self.r_pu[(fbus, tbus)] = r
                self.x_pu[(fbus, tbus)] = x
                idx += 1
            self.n_branches = len(self.branch_list)

        # Ensure symmetric branch index lookup
        for (fbus, tbus), idx in list(self.branch_idx.items()):
            self.branch_idx[(tbus, fbus)] = idx
            self.r_pu[(tbus, fbus)] = self.r_pu[(fbus, tbus)]
            self.x_pu[(tbus, fbus)] = self.x_pu[(fbus, tbus)]

    def _parse_matrix_rows(self, text):
        """Parse a MATLAB matrix into list of numeric rows. Handles semicolons."""
        rows = []
        current = text.strip()
        row_texts = re.split(r";", current)
        for rt in row_texts:
            rt = rt.strip()
            if not rt:
                continue
            # Handle MATLAB continuation lines (no semicolon, just whitespace)
            vals = []
            for token in rt.split():
                try:
                    vals.append(float(token))
                except ValueError:
                    pass
            if vals:
                rows.append(vals)
        return rows

    def _build_topology(self):
        """Build radial topology using only normally-closed branches (status=1)."""
        adj = {i: [] for i in range(1, self.n_buses + 1)}
        for (fbus, tbus, r, x, status) in self.branch_list:
            if status == 1:  # normally closed
                adj[fbus].append(tbus)
                adj[tbus].append(fbus)

        # BFS from slack bus (1) to determine parent-child
        visited = set()
        queue = deque([1])
        visited.add(1)
        self.parent[1] = 0  # slack has no parent
        self.children = {i: [] for i in range(1, self.n_buses + 1)}

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    self.parent[v] = u
                    self.children[u].append(v)
                    queue.append(v)

    def _compute_downstream_sets(self):
        """Compute set of all buses in the subtree rooted at each bus (post-order)."""
        self.downstream = {i: set() for i in range(1, self.n_buses + 1)}

        def dfs(u):
            ds = {u}
            for v in self.children[u]:
                ds |= dfs(v)
            self.downstream[u] = ds
            return ds

        dfs(1)

    def get_closed_branches(self):
        """Return list of (fbus, tbus, r_pu, x_pu) for normally-closed branches."""
        result = []
        for (fbus, tbus, r, x, status) in self.branch_list:
            if status == 1:
                result.append((fbus, tbus, r, x))
        return result

    def get_leaf_buses(self):
        """Return leaf buses (no children) beyond slack."""
        return [i for i in range(2, self.n_buses + 1) if not self.children[i]]

    def print_summary(self):
        """Print network summary statistics."""
        print("=" * 60)
        print("IEEE 33-Bus Distribution System Summary")
        print("=" * 60)
        print(f"  Buses:    {self.n_buses}")
        print(f"  Branches: {self.n_branches} ({len(self.get_closed_branches())} closed, "
              f"{self.n_branches - len(self.get_closed_branches())} tie switches)")
        total_p = sum(d["Pd"] for d in self.bus_data.values())
        total_q = sum(d["Qd"] for d in self.bus_data.values())
        print(f"  Total load: {total_p:.4f} MW, {total_q:.4f} MVAr")
        print(f"  Base: {self.S_base} MVA, {self.V_base} kV")
        print(f"  Z_base: {self.Z_base:.4f} Ohm")
        print(f"  Slck bus: 1")
        print(f"  Leaf buses: {self.get_leaf_buses()}")
        print("=" * 60)


if __name__ == "__main__":
    net = Network("data/case33bw.m")
    net.print_summary()
    print("\nTopology (bus -> parent):")
    for i in range(2, net.n_buses + 1):
        print(f"  Bus {i:2d} -> parent {net.parent[i]:2d},  "
              f"downstream: {len(net.downstream[i])} buses")
