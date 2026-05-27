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
