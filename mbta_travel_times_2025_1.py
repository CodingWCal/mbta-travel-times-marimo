"""MBTA travel times (2025) — Marimo notebook.

Run: ``marimo edit mbta_travel_times_2025.py``
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    return mo.md(
        r"""
# Same city, different pace

**Question:** On weekdays, which MBTA line’s stop-to-stop trips tend to run **slowest compared to that line’s own “normal” for the same corridor**—and **when** does that show up?

**Why travel times:** Headways capture *waiting*. Travel-time rows capture *moving*: each record is one trip’s time from a departure at stop A to an arrival at the next stop B.

**Method:** For every `(route, A→B)` segment with enough observations, we set a **segment baseline** = median travel time for that segment in the month. Each trip’s **excess time** = actual minus baseline—so long physical segments do not automatically make a line “look bad.”
"""
    )


@app.cell
def __(mo):
    from pathlib import Path

    # Change if your CSVs live elsewhere.
    DATA_DIR = Path(r"C:\Users\calvi\Downloads\TravelTimes_2025\TravelTimes_2025")

    month_ui = mo.ui.dropdown(
        options=[f"2025-{m:02d}" for m in range(1, 13)],
        value="2025-10",
        label="Month",
    )

    mode_ui = mo.ui.dropdown(
        options=["Heavy rail (HR)", "Light rail (LR)"],
        value="Heavy rail (HR)",
        label="Mode",
    )

    return DATA_DIR, mode_ui, month_ui


@app.cell
def __():
    # Why: Marimo runs like a server; Agg avoids GUI backends that sometimes produce blank figures.
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 10,
        }
    )

    return np, pd, plt


@app.cell
def __(DATA_DIR, mode_ui, month_ui, mo, np, pd, plt):
    suffix = "HRTravelTimes.csv" if mode_ui.value.endswith("(HR)") else "LRTravelTimes.csv"
    csv_path = DATA_DIR / f"{month_ui.value}_{suffix}"

    if not csv_path.exists():
        return mo.md(
            f"### Missing data file\n\nExpected `{csv_path.name}` under `{DATA_DIR}`.\n\n"
            "Unzip **Travel Times 2025** from the MBTA portal into that folder, or update `DATA_DIR` in the notebook."
        )

    usecols = [
        "route_id",
        "from_parent_station",
        "to_parent_station",
        "from_stop_departure_datetime",
        "travel_time_sec",
    ]

    # Only columns needed for the narrative + charts (keeps memory lower on big files).
    raw = pd.read_csv(csv_path, usecols=usecols)

    df = raw.dropna(subset=["travel_time_sec", "from_stop_departure_datetime"]).copy()
    df = df[(df["travel_time_sec"] > 0) & (df["travel_time_sec"] < 3600)]

    df["segment"] = (
        df["route_id"].astype(str)
        + "|"
        + df["from_parent_station"].astype(str)
        + "→"
        + df["to_parent_station"].astype(str)
    )

    seg_counts = df.groupby("segment", observed=True)["travel_time_sec"].transform("count")
    # Rare segments => unstable medians; 30 is a pragmatic minimum for a short project.
    df = df[seg_counts >= 30].copy()

    baseline = df.groupby("segment", observed=True)["travel_time_sec"].transform("median")
    df["excess_sec"] = df["travel_time_sec"] - baseline

    dt = pd.to_datetime(df["from_stop_departure_datetime"], utc=True)
    local = dt.dt.tz_convert("America/New_York")
    # Why: presentation-friendly axis (Boston/Cambridge audience) instead of raw UTC hours.
    df["hour_local"] = local.dt.hour
    df["dow"] = local.dt.dayofweek
    df["is_weekend"] = df["dow"] >= 5

    weekday = df[~df["is_weekend"]].copy()
    route_order = sorted(weekday["route_id"].dropna().unique().tolist())

    pivot = weekday.pivot_table(
        index="route_id",
        columns="hour_local",
        values="excess_sec",
        aggfunc="median",
    ).reindex(route_order)
    # Keep a stable 0–23 axis even if some local hours are missing in the extract.
    pivot = pivot.reindex(columns=list(range(24)))

    # --- Figure 1 ---
    fig1, ax1 = plt.subplots(figsize=(11, 3.8))
    heat = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    im = ax1.imshow(
        heat,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels(list(pivot.index))
    ax1.set_xlabel("Hour of day (America/New_York)")
    ax1.set_ylabel("Route")
    ax1.set_title(
        "Median excess travel time vs segment baseline — weekdays\n"
        "(positive = slower than that segment’s typical month median)"
    )
    ax1.set_xticks(range(0, 24))
    ax1.set_xticklabels([str(h) for h in range(24)])
    cbar = fig1.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("Median excess (seconds)")
    fig1.tight_layout()

    # --- Figure 2 ---
    pm = weekday[weekday["hour_local"].between(17, 19)]
    pm_summary = (
        pm.groupby("route_id", observed=True)["excess_sec"]
        .agg(
            median_excess_sec="median",
            p75_excess_sec=lambda s: float(s.quantile(0.75)),
        )
        .reindex(route_order)
    )

    route_colors = {
        "Red": "#DA291C",
        "Orange": "#ED8B00",
        "Blue": "#003DA5",
        "Green": "#00843D",
        "Green-B": "#00843D",
        "Green-C": "#00843D",
        "Green-D": "#00843D",
        "Green-E": "#00843D",
        "Mattapan": "#DA291C",
    }
    colors = [route_colors.get(str(r), "#457B9D") for r in pm_summary.index]

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(pm_summary))
    med = pm_summary["median_excess_sec"].to_numpy(dtype=float)
    p75 = pm_summary["p75_excess_sec"].to_numpy(dtype=float)
    ax2.bar(x, med, color=colors, alpha=0.92, label="Weekday 5–7pm median excess (sec)")
    ax2.vlines(x, med, p75, colors="#111111", linewidth=2, alpha=0.55, label="Up to 75th pct (tail pain)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(pm_summary.index), rotation=0)
    ax2.set_ylabel("Excess travel time (seconds)")
    ax2.set_title("Weekday evening peak (5–7pm ET): slowdown vs typical hops")
    ax2.legend(loc="upper right", frameon=False)
    ax2.axhline(0, color="black", linewidth=1)
    fig2.tight_layout()

    worst_route = pm_summary["median_excess_sec"].idxmax()
    worst_median_sec = float(pm_summary.loc[worst_route, "median_excess_sec"])

    pitch = (
        "### 60-second presentation script\n\n"
        "**0:00–0:15 — Hook:** Boston riders argue Red vs Orange, but those fights often mix up "
        "*waiting* vs *moving*. I used MBTA **travel times**: each row is a real stop-to-stop hop.\n\n"
        "**0:15–0:35 — Heatmap:** Each hop is compared to **its own typical time that month**. "
        "The heatmap shows **weekday median excess** by **route** and **hour** for **"
        f"{month_ui.value}**.\n\n"
        "**0:35–0:50 — Bars:** Zooming to **weeknight rush** (5–7pm ET), **"
        f"{worst_route}** has the largest median excess here—about **{worst_median_sec:.0f} seconds**.\n\n"
        "**0:50–1:00 — Caveat + punchline:** This view uses **America/New_York** hours from the MBTA timestamps. "
        "Still, the idea is simple: **slowdown is not evenly spread across lines or across the day**, "
        "even after comparing each corridor to itself."
    )

    return mo.vstack(
        [
            mo.md(
                "### Dataset overview\n"
                f"- **File:** `{csv_path.name}`\n"
                "- **Portal:** [MBTA historical performance data](https://www.mbta.com/developers/historical-performance-data)\n"
                "- **Row:** one observed **stop-to-stop** `travel_time_sec` for a trip.\n"
            ),
            mo.md(
                "### Exploration (lightweight)\n"
                f"- **Rows after QC:** {len(df):,}\n"
                f"- **Routes:** {', '.join(route_order)}\n"
                f"- **Weekday / weekend rows:** {len(weekday):,} / {int(df['is_weekend'].sum()):,}\n"
            ),
            mo.md(
                "### Visualizations\n"
                "If charts do not appear after a reload: confirm you opened **this** notebook file in marimo, "
                "and that `matplotlib` is installed in the same environment running `marimo edit ...`."
            ),
            mo.mpl(fig1),
            mo.mpl(fig2),
            mo.md(
                "### Key insight\n"
                f"In **{month_ui.value}** weekday evenings (**5–7pm ET**), **{worst_route}** "
                f"shows the largest **median excess** vs segment baselines (**~{worst_median_sec:.0f}s**). "
                "That is a *relative slowdown* signal—not geographic line length."
            ),
            mo.md(
                "### Caveats\n"
                "- Hours are **America/New_York** derived from MBTA `Z` timestamps.\n"
                "- MBTA warns extracts can be incomplete around data issues.\n"
                "- Baseline is the **same-month** segment median (not vs published schedules).\n"
            ),
            mo.md(
                "### Next iteration (if you have another hour)\n"
                "Split **inbound vs outbound**, and anchor on one famous corridor (e.g. Harvard→Central) "
                "without needing a map.\n"
            ),
            mo.md(pitch),
        ],
        gap=1,
        align="start",
    )

