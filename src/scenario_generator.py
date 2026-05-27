"""Generate time-series load, PV, wind profiles and cluster into typical days."""

import numpy as np
from sklearn.cluster import KMeans

# System base for per-unit conversion
S_BASE = 10.0  # MVA

# PV buses: spread across the feeder (mid and end points)
PV_BUSES = [6, 12, 15, 22, 25, 30, 18, 33]
PV_TOTAL_KW = 2500.0   # 2.5 MW total PV (~67% of peak load)

# Wind buses: near feeder ends
WIND_BUSES = [18, 33]
WIND_TOTAL_KW = 1500.0  # 1.5 MW total wind (~40% of peak load)

# Electricity price time-of-use (cents/kWh)
PRICE_SUMMER_PEAK = 0.18
PRICE_SUMMER_OFF = 0.09
PRICE_WINTER_PEAK = 0.13
PRICE_WINTER_OFF = 0.07
PRICE_SPRING_FALL = 0.08


def _generate_diurnal_pattern():
    """Generate 24h diurnal load pattern (two peaks: morning ~10h, evening ~19h)."""
    h = np.arange(24)
    # Base pattern: residential + light commercial
    pattern = 0.65 + 0.15 * np.sin(np.pi * (h - 6) / 12) \
              + 0.20 * np.exp(-((h - 10) ** 2) / 8) \
              + 0.25 * np.exp(-((h - 19) ** 2) / 8)
    pattern = pattern / pattern.max()  # normalize peak = 1.0
    return pattern


def _seasonal_factor(month):
    """Seasonal load multiplier by month."""
    # Summer (Jun-Aug): 1.0-1.15, Winter (Dec-Feb): 0.85-0.95, Spring/Fall: 0.70-0.85
    if month in [6, 7, 8]:
        return 1.10
    elif month in [12, 1, 2]:
        return 0.88
    elif month in [3, 4, 5]:
        return 0.78
    else:  # Sep, Oct, Nov
        return 0.82


def _get_pv_profile(day_of_year, cloud_factor, n_hours=24):
    """Generate hourly PV capacity factor for a given day and cloud condition."""
    hours = np.arange(n_hours)

    # Sunlight hours vary by season (simplified model)
    # Day length: ~10h winter, ~14h summer (mid-latitudes)
    day_length = 10.0 + 4.0 * np.sin(np.pi * (day_of_year - 80) / 365) ** 2

    # Sunrise time centered at solar noon (12:00)
    sunrise = 12.0 - day_length / 2
    sunset = 12.0 + day_length / 2

    # Ideal clear-sky profile: trapezoidal
    cf = np.zeros(n_hours)
    for h in range(n_hours):
        t_mid = h + 0.5
        if t_mid <= sunrise or t_mid >= sunset:
            cf[h] = 0
        else:
            # Rise, plateau, fall
            if t_mid < sunrise + 1.5:
                cf[h] = (t_mid - sunrise) / 1.5
            elif t_mid > sunset - 1.5:
                cf[h] = (sunset - t_mid) / 1.5
            else:
                # Noon peak with slight cloud attenuation varies by season
                cf[h] = 0.85 + 0.15 * np.sin(np.pi * (day_of_year - 80) / 365)

    cf = np.clip(cf, 0, 1)
    # Apply cloud cover (daily random factor)
    cf *= cloud_factor
    return np.clip(cf, 0, 1)


def _get_wind_speed(day_of_year, n_hours=24):
    """Generate hourly wind speed (m/s) using Weibull + AR(1) process."""
    # Seasonal mean wind speed
    season_mean = 7.0 + 1.5 * np.sin(2 * np.pi * (day_of_year - 30) / 365)

    # AR(1) process with Weibull base
    rho = 0.85
    sigma = 2.5
    wind = np.zeros(n_hours + 24)
    # Initialize with previous day's steady state
    wind[0] = season_mean + np.random.randn() * sigma * np.sqrt(1 - rho**2)
    for h in range(1, n_hours + 24):
        wind[h] = season_mean + rho * (wind[h - 1] - season_mean) \
                  + sigma * np.sqrt(1 - rho**2) * np.random.randn()

    wind = wind[24:]  # discard spin-up
    # Apply Weibull transformation
    wind = np.clip(wind, 0.5, 25)
    return wind


def _wind_power(wind_speed, rated_speed=12.0, cut_in=3.0, cut_out=22.0):
    """Convert wind speed (m/s) to power output (capacity factor)."""
    cf = np.zeros_like(wind_speed)
    mask = (wind_speed >= cut_in) & (wind_speed < rated_speed)
    cf[mask] = (wind_speed[mask] - cut_in) / (rated_speed - cut_in)
    cf[(wind_speed >= rated_speed) & (wind_speed < cut_out)] = 1.0
    cf = np.clip(cf, 0, 1)
    return cf


def generate_profiles(net, n_days=365, seed=42):
    """
    Generate 365 days x 24h load, PV, wind profiles.

    Returns dict with keys:
        load_MW[n_buses, n_days*24]
        pv_MW[n_buses, n_days*24]
        wind_MW[n_buses, n_days*24]
        price[n_days*24]           electricity price ($/kWh)
        pv_buses, wind_buses
        n_days, n_hours_per_day
    """
    np.random.seed(seed)
    n_buses = net.n_buses
    n_hours = 24
    n_total = n_days * n_hours

    diurnal = _generate_diurnal_pattern()
    base_load = np.array([net.bus_data[i]["Pd"] for i in range(1, n_buses + 1)])

    load_MW = np.zeros((n_buses, n_total))
    pv_MW = np.zeros((n_buses, n_total))
    wind_MW = np.zeros((n_buses, n_total))
    price = np.zeros(n_total)

    # Distribute PV capacity across PV buses
    pv_ratios = np.array([0.25, 0.10, 0.08, 0.12, 0.15, 0.10, 0.10, 0.10])  # must sum to 1
    pv_kw_per_bus = dict(zip(PV_BUSES, PV_TOTAL_KW * pv_ratios))

    # Distribute wind capacity
    wind_ratios = np.array([0.50, 0.50])
    wind_kw_per_bus = dict(zip(WIND_BUSES, WIND_TOTAL_KW * wind_ratios))

    for d in range(n_days):
        # Determine month from day index
        month = ((d % 365) // 30) + 1
        if month > 12:
            month = 12

        day_of_year = (d % 365) + 1
        is_weekend = (d % 7) in [5, 6]

        # Load profile for this day
        seas = _seasonal_factor(month)
        weekend_factor = 0.85 if is_weekend else 1.0
        daily_noise = 1.0 + 0.03 * np.random.randn()
        base_hourly = diurnal * seas * weekend_factor * daily_noise

        for h in range(n_hours):
            t = d * n_hours + h
            load_MW[:, t] = base_load * base_hourly[h]

        # PV profile
        # Cloud factor: some days clear, some cloudy, some overcast
        cloud_rand = np.random.rand()
        if cloud_rand < 0.4:
            cloud_factor = 0.9 + 0.1 * np.random.rand()   # mostly clear
        elif cloud_rand < 0.7:
            cloud_factor = 0.5 + 0.3 * np.random.rand()   # partly cloudy
        elif cloud_rand < 0.9:
            cloud_factor = 0.2 + 0.2 * np.random.rand()   # mostly cloudy
        else:
            cloud_factor = 0.05 + 0.1 * np.random.rand()  # overcast

        pv_cf = _get_pv_profile(day_of_year, cloud_factor)

        for h in range(n_hours):
            t = d * n_hours + h
            for bus, kw in pv_kw_per_bus.items():
                pv_MW[bus - 1, t] = kw / 1000.0 * pv_cf[h]

        # Wind profile
        wind_speed = _get_wind_speed(day_of_year)
        wind_cf = _wind_power(wind_speed)

        for h in range(n_hours):
            t = d * n_hours + h
            for bus, kw in wind_kw_per_bus.items():
                wind_MW[bus - 1, t] = kw / 1000.0 * wind_cf[h]

        # Electricity price
        for h in range(n_hours):
            t = d * n_hours + h
            is_peak = (9 <= h <= 11) or (17 <= h <= 20)
            if month in [6, 7, 8]:
                price[t] = PRICE_SUMMER_PEAK if is_peak else PRICE_SUMMER_OFF
            elif month in [12, 1, 2]:
                price[t] = PRICE_WINTER_PEAK if is_peak else PRICE_WINTER_OFF
            else:
                price[t] = PRICE_SPRING_FALL

    return {
        "load_MW": load_MW,
        "pv_MW": pv_MW,
        "wind_MW": wind_MW,
        "price_per_kwh": price,
        "pv_buses": PV_BUSES,
        "wind_buses": WIND_BUSES,
        "n_days": n_days,
        "n_hours_per_day": n_hours,
    }


def cluster_typical_days(profiles, n_clusters=6, seed=42):
    """
    Cluster 365 daily load+PV+wind patterns into K typical days.

    Returns dict with:
        centroids: (n_clusters, n_hours) per profile type
        weights: (n_clusters,) fraction of days in each cluster
        labels: (n_days,) cluster assignment per day
        n_clusters, n_hours_per_day
    """
    n_days = profiles["n_days"]
    n_hours = profiles["n_hours_per_day"]

    # Build feature matrix: each day is [24h total load + 24h total PV + 24h total wind]
    features = np.zeros((n_days, n_hours * 3))
    for d in range(n_days):
        t0 = d * n_hours
        t1 = t0 + n_hours
        total_load = profiles["load_MW"].sum(axis=0)[t0:t1]
        total_pv = profiles["pv_MW"].sum(axis=0)[t0:t1]
        total_wind = profiles["wind_MW"].sum(axis=0)[t0:t1]

        # Normalize each
        if total_load.max() > 0:
            total_load = total_load / total_load.max()
        if total_pv.max() > 0:
            total_pv = total_pv / total_pv.max()
        if total_wind.max() > 0:
            total_wind = total_wind / total_wind.max()

        features[d, :n_hours] = total_load
        features[d, n_hours:2 * n_hours] = total_pv
        features[d, 2 * n_hours:] = total_wind

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(features)

    # Compute cluster weights and centroids
    weights = np.zeros(n_clusters)
    for k in range(n_clusters):
        weights[k] = np.sum(labels == k) / n_days

    # Extract actual centroids (mean of each cluster) for load, PV, wind
    centroids_load = np.zeros((n_clusters, n_hours))
    centroids_pv = np.zeros((n_clusters, n_hours))
    centroids_wind = np.zeros((n_clusters, n_hours))
    centroids_price = np.zeros((n_clusters, n_hours))

    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        for d in idx:
            t0 = d * n_hours
            t1 = t0 + n_hours
            centroids_load[k] += profiles["load_MW"].sum(axis=0)[t0:t1]
            centroids_pv[k] += profiles["pv_MW"].sum(axis=0)[t0:t1]
            centroids_wind[k] += profiles["wind_MW"].sum(axis=0)[t0:t1]
            centroids_price[k] += profiles["price_per_kwh"][t0:t1]
        centroids_load[k] /= len(idx)
        centroids_pv[k] /= len(idx)
        centroids_wind[k] /= len(idx)
        centroids_price[k] /= len(idx)

    return {
        "n_clusters": n_clusters,
        "n_hours_per_day": n_hours,
        "weights": weights,
        "labels": labels,
        "load_centroids": centroids_load,     # (K, 24) total system load, MW
        "pv_centroids": centroids_pv,          # (K, 24) total system PV, MW
        "wind_centroids": centroids_wind,      # (K, 24) total system wind, MW
        "price_centroids": centroids_price,    # (K, 24) $/kWh
    }


def build_scenario_data(profiles, typical_days, net):
    """
    Build full bus-level time-series data for each typical day.

    Returns:
        scenarios: list of dicts, one per typical day, each with:
            weight, load_pu[bus, h], pv_pu[bus, h], wind_pu[bus, h], price[h]
            total_hours
    """
    n_clusters = typical_days["n_clusters"]
    n_hours = typical_days["n_hours_per_day"]
    n_buses = net.n_buses

    # Map cluster centroids back to individual bus profiles
    # We need the bus-level breakdown per cluster centroid
    # Approach: find the day closest to each centroid, use that day's bus-level data
    # Better approach: compute bus-level centroids directly

    # Recompute bus-level centroids
    bus_load_centroids = np.zeros((n_buses, n_clusters, n_hours))
    bus_pv_centroids = np.zeros((n_buses, n_clusters, n_hours))
    bus_wind_centroids = np.zeros((n_buses, n_clusters, n_hours))
    bus_price_centroids = np.zeros((n_clusters, n_hours))

    for k in range(n_clusters):
        idx = np.where(typical_days["labels"] == k)[0]
        if len(idx) == 0:
            continue
        for d in idx:
            t0 = d * n_hours
            t1 = t0 + n_hours
            bus_load_centroids[:, k, :] += profiles["load_MW"][:, t0:t1]
            bus_pv_centroids[:, k, :] += profiles["pv_MW"][:, t0:t1]
            bus_wind_centroids[:, k, :] += profiles["wind_MW"][:, t0:t1]
            bus_price_centroids[k, :] += profiles["price_per_kwh"][t0:t1]
        bus_load_centroids[:, k, :] /= len(idx)
        bus_pv_centroids[:, k, :] /= len(idx)
        bus_wind_centroids[:, k, :] /= len(idx)
        bus_price_centroids[k, :] /= len(idx)

    # Convert to per-unit (base = 10 MVA)
    scenarios = []
    for k in range(n_clusters):
        load_pu = bus_load_centroids[:, k, :] / net.S_base
        pv_pu = bus_pv_centroids[:, k, :] / net.S_base
        wind_pu = bus_wind_centroids[:, k, :] / net.S_base

        # Also compute reactive load (assume PF = 0.85 lagging)
        base_q_mw = np.array([net.bus_data[i]["Qd"] for i in range(1, n_buses + 1)])
        # Scale reactive load proportionally to active load
        q_multipliers = np.zeros((n_buses, n_hours))
        for bus in range(n_buses):
            if bus_load_centroids[bus, k, :].max() > 0:
                # Use ratio of hourly load to base load
                base_p = net.bus_data[bus + 1]["Pd"]
                if base_p > 0:
                    q_multipliers[bus, :] = bus_load_centroids[bus, k, :] / base_p
                else:
                    q_multipliers[bus, :] = 0
        q_load_pu = np.zeros((n_buses, n_hours))
        for bus in range(n_buses):
            q_load_pu[bus, :] = base_q_mw[bus] * q_multipliers[bus, :] / net.S_base

        scenarios.append({
            "weight": typical_days["weights"][k],
            "load_pu": load_pu,           # (n_buses, n_hours)
            "q_load_pu": q_load_pu,       # (n_buses, n_hours)
            "pv_pu": pv_pu,               # (n_buses, n_hours)
            "wind_pu": wind_pu,           # (n_buses, n_hours)
            "price_per_kwh": bus_price_centroids[k, :],  # (n_hours,)
            "n_hours": n_hours,
        })

    return scenarios


def select_dro_samples(profiles, N=20, seed=42):
    """
    Select N representative daily samples via K-means representative selection.

    Uses K-means to cluster days, then picks the day closest to each centroid
    (preserving original data rather than using averaged centroids).

    Returns:
        sample_indices: list of N day indices (0 to n_days-1)
        selection: dict with keys load_total, pv_total, wind_total, price_24h,
                   each (N, 24) arrays of system-level data for selected days
    """
    np.random.seed(seed)
    n_days = profiles["n_days"]
    n_hours = profiles["n_hours_per_day"]

    # Build feature matrix for clustering
    features = np.zeros((n_days, n_hours * 3))
    for d in range(n_days):
        t0 = d * n_hours
        t1 = t0 + n_hours
        total_load = profiles["load_MW"].sum(axis=0)[t0:t1]
        total_pv = profiles["pv_MW"].sum(axis=0)[t0:t1]
        total_wind = profiles["wind_MW"].sum(axis=0)[t0:t1]
        if total_load.max() > 0:
            total_load = total_load / total_load.max()
        if total_pv.max() > 0:
            total_pv = total_pv / total_pv.max()
        if total_wind.max() > 0:
            total_wind = total_wind / total_wind.max()
        features[d, :n_hours] = total_load
        features[d, n_hours:2 * n_hours] = total_pv
        features[d, 2 * n_hours:] = total_wind

    # K-means to N clusters
    kmeans = KMeans(n_clusters=N, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(features)

    # Select the day closest to each centroid
    sample_indices = []
    centroids = kmeans.cluster_centers_
    for k in range(N):
        cluster_days = np.where(labels == k)[0]
        if len(cluster_days) == 0:
            continue
        # Find day in cluster closest to centroid
        dist = np.sum((features[cluster_days] - centroids[k]) ** 2, axis=1)
        best_day = cluster_days[np.argmin(dist)]
        sample_indices.append(best_day)

    sample_indices = sorted(sample_indices)

    # Extract system-level data for selected days
    load_total = np.zeros((N, n_hours))
    pv_total = np.zeros((N, n_hours))
    wind_total = np.zeros((N, n_hours))
    price_24h = np.zeros((N, n_hours))

    for k, d in enumerate(sample_indices):
        t0 = d * n_hours
        t1 = t0 + n_hours
        load_total[k] = profiles["load_MW"].sum(axis=0)[t0:t1]
        pv_total[k] = profiles["pv_MW"].sum(axis=0)[t0:t1]
        wind_total[k] = profiles["wind_MW"].sum(axis=0)[t0:t1]
        price_24h[k] = profiles["price_per_kwh"][t0:t1]

    return sample_indices, {
        "load_total": load_total,       # (N, 24) MW
        "pv_total": pv_total,           # (N, 24) MW
        "wind_total": wind_total,       # (N, 24) MW
        "price_24h": price_24h,         # (N, 24) $/kWh
        "sample_indices": sample_indices,
        "n_samples": N,
    }


def compute_wasserstein_radius(selection, beta=0.10):
    """
    Compute Wasserstein ball radius ε for ∞-norm metric.

    Formula: ε = σ_ref · (log(1/β) / N)^{1/d}
    where σ_ref = average range across uncertainty dimensions from sample data,
    N = number of samples, d = uncertainty dimension.

    Args:
        selection: dict from select_dro_samples
        beta: confidence level (default 0.1 for 90% confidence)

    Returns:
        epsilon: radius value (in MW, same units as system-level data)
        epsilon_per_dim: (3*24,) array of per-dimension radius (for diagnostic)
    """
    N = selection["n_samples"]

    # Stack all uncertainty dimensions: (N, 72) where 72 = 3 × 24h
    data = np.hstack([
        selection["load_total"],   # (N, 24)
        selection["pv_total"],     # (N, 24)
        selection["wind_total"],   # (N, 24)
    ])

    d = data.shape[1]  # 72

    # σ_ref: average range (max-min) across all dimensions
    ranges = data.max(axis=0) - data.min(axis=0)
    sigma_ref = np.mean(ranges)

    # Wasserstein radius formula
    # ε = σ_ref * (log(1/β) / N)^{1/max(d,2)}
    d_eff = max(d, 2)
    eps = sigma_ref * (np.log(1.0 / beta) / N) ** (1.0 / d_eff)

    # Per-dimension radius (for visualization)
    eps_per_dim = ranges * (np.log(1.0 / beta) / N) ** (1.0 / d_eff)

    return eps, eps_per_dim


def build_dro_scenario_data(net, profiles, N=20, beta=0.10, seed=42):
    """
    Build DRO scenario dataset with pre-computed worst-case bus-level data.

    For ∞-norm Wasserstein DRO with monotonic cost (more load / less generation
    = higher cost), the worst-case perturbation within ||Δ||_∞ ≤ ε is:
      - Δ_load = +ε  (more load)
      - Δ_pv   = -ε   (less PV, clamped at 0)
      - Δ_wind = -ε   (less wind, clamped at 0)

    Bus-level values are allocated from perturbed system totals via fixed ratios.

    Returns a dict with:
        scenarios: list of N dicts (compatible with planning_model.build_and_solve)
            weight, load_pu[bus, h], pv_pu[bus, h], wind_pu[bus, h],
            q_load_pu[bus, h], price_per_kwh[h], n_hours,
            load_nominal_total[24], pv_nominal_total[24], wind_nominal_total[24],
            load_wc_total[24], pv_wc_total[24], wind_wc_total[24]
        epsilon: Wasserstein radius (MW)
        epsilon_per_dim: per-dimension radius
        sample_indices: selected day indices
        N, beta
    """
    n_buses = net.n_buses
    n_hours = profiles["n_hours_per_day"]

    # Select samples
    sample_indices, selection = select_dro_samples(profiles, N=N, seed=seed)

    # Compute Wasserstein radius
    eps, eps_per_dim = compute_wasserstein_radius(selection, beta=beta)

    # Pre-compute allocation ratios
    base_load = np.array([net.bus_data[i]["Pd"] for i in range(1, n_buses + 1)])
    total_base_load = base_load.sum()
    load_ratio = base_load / total_base_load if total_base_load > 0 else np.ones(n_buses) / n_buses

    # PV allocation ratio from installed capacities
    pv_buses = profiles["pv_buses"]
    wind_buses = profiles["wind_buses"]
    pv_total_kw = PV_TOTAL_KW
    wind_total_kw = WIND_TOTAL_KW

    pv_ratios_list = np.array([0.25, 0.10, 0.08, 0.12, 0.15, 0.10, 0.10, 0.10])
    pv_kw_per_bus = dict(zip(pv_buses, pv_total_kw * pv_ratios_list))
    pv_ratio = np.zeros(n_buses)
    for bus, kw in pv_kw_per_bus.items():
        pv_ratio[bus - 1] = kw / pv_total_kw if pv_total_kw > 0 else 0

    wind_ratios_list = np.array([0.50, 0.50])
    wind_kw_per_bus = dict(zip(wind_buses, wind_total_kw * wind_ratios_list))
    wind_ratio = np.zeros(n_buses)
    for bus, kw in wind_kw_per_bus.items():
        wind_ratio[bus - 1] = kw / wind_total_kw if wind_total_kw > 0 else 0

    # Per-bus Q/P ratio from base data
    base_q = np.array([net.bus_data[i]["Qd"] for i in range(1, n_buses + 1)])
    q_factor = np.divide(base_q, base_load, where=base_load > 1e-9,
                         out=np.zeros_like(base_q))

    # Build scenarios with worst-case bus-level p.u. data
    scenarios = []
    for k, d in enumerate(sample_indices):
        load_nom = selection["load_total"][k]      # (24,) MW
        pv_nom = selection["pv_total"][k]          # (24,) MW
        wind_nom = selection["wind_total"][k]      # (24,) MW
        price = selection["price_24h"][k]          # (24,) $/kWh

        # Worst-case system-level values (∞-norm adversarial perturbation)
        load_wc = load_nom + eps                         # +ε: more load
        pv_wc = np.maximum(0, pv_nom - eps)             # -ε: less PV, floor 0
        wind_wc = np.maximum(0, wind_nom - eps)         # -ε: less wind, floor 0

        # Allocate to buses and convert to p.u.
        load_pu = np.zeros((n_buses, n_hours))
        pv_pu = np.zeros((n_buses, n_hours))
        wind_pu = np.zeros((n_buses, n_hours))
        q_load_pu = np.zeros((n_buses, n_hours))

        for bus in range(n_buses):
            load_pu[bus, :] = load_ratio[bus] * load_wc / net.S_base
            pv_pu[bus, :] = pv_ratio[bus] * pv_wc / net.S_base
            wind_pu[bus, :] = wind_ratio[bus] * wind_wc / net.S_base
            q_load_pu[bus, :] = q_factor[bus] * load_pu[bus, :]

        scenarios.append({
            "weight": 1.0 / N,                  # equal weight for all samples
            "load_pu": load_pu,                 # (n_buses, 24) worst-case
            "q_load_pu": q_load_pu,             # (n_buses, 24) worst-case
            "pv_pu": pv_pu,                     # (n_buses, 24) worst-case
            "wind_pu": wind_pu,                 # (n_buses, 24) worst-case
            "price_per_kwh": price,             # (24,)
            "n_hours": n_hours,
            "load_nominal_total": load_nom,     # for reporting
            "pv_nominal_total": pv_nom,
            "wind_nominal_total": wind_nom,
            "load_wc_total": load_wc,
            "pv_wc_total": pv_wc,
            "wind_wc_total": wind_wc,
            "day_index": int(d),
        })

    return {
        "scenarios": scenarios,
        "epsilon": eps,
        "epsilon_per_dim": eps_per_dim,
        "sample_indices": sample_indices,
        "N": N,
        "beta": beta,
    }


if __name__ == "__main__":
    from parse_network import Network
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = Network("data/case33bw.m")
    profiles = generate_profiles(net, n_days=365)
    typical = cluster_typical_days(profiles, n_clusters=6)

    # Plot typical days
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for k in range(min(6, typical["n_clusters"])):
        ax = axes[k // 2, k % 2]
        h = np.arange(24)
        ax.plot(h, typical["load_centroids"][k], "b-", label="Load (MW)")
        ax.plot(h, typical["pv_centroids"][k], "r-", label="PV (MW)")
        ax.plot(h, typical["wind_centroids"][k], "g-", label="Wind (MW)")
        ax.set_title(f"Day type {k+1} (weight={typical['weights'][k]:.3f})")
        ax.set_xlabel("Hour")
        ax.set_ylabel("MW")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/typical_days.png", dpi=150)
    print("Saved typical days plot to results/typical_days.png")
    print(f"Cluster weights: {typical['weights']}")
