"""OAU campus transport study: report and interactive dashboard (Streamlit).

Run locally with ``streamlit run app.py``. Every data file sits next to this
file; team credits come from ``team.csv`` (edit it to change names/roles).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import re
import types

# ==========================================================================
# DATA LOADING AND CLEANING (formerly cng_data.py)
# ==========================================================================

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data helpers (copied from the project's src/ package so this file is standalone)
# ---------------------------------------------------------------------------
import math
from datetime import datetime, time
from typing import Any

HERE = Path(__file__).resolve().parent
CAPACITIES = {"CNG bus": 13, "Keke": 5}
SHEET_CONFIG = {
    "queue": {"csv": "B1_Queue_and_Demand.csv", "event_time": "Time"},
    "wait": {"csv": "B2_Wait_Sample.csv", "event_time": "Time joined"},
    "vehicle": {"csv": "C_Vehicle_and_Trip.csv", "event_time": "Arrived"},
}


def find_file(name: str) -> Path:
    """Locate a data file next to app.py, tolerating renamed copies such as
    'name_1.csv' or 'name (1).csv' that browsers create on repeated downloads."""
    folder = Path(__file__).resolve().parent
    exact = folder / name
    if exact.exists():
        return exact
    stem, ext = Path(name).stem.lower(), Path(name).suffix.lower()
    pattern = re.compile(rf"^{re.escape(stem)}(?:[ _-]?\(?\d+\)?)?{re.escape(ext)}$")
    matches = sorted((f for f in folder.iterdir() if f.is_file() and pattern.match(f.name.lower())),
                     key=lambda f: f.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(name)


def read_csv_sheet(name: str) -> pd.DataFrame:
    path = find_file(name)
    try:
        frame = pd.read_csv(path, header=3, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, header=3, encoding="cp1252")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.dropna(how="all").reset_index(drop=True)


def load_all_data() -> dict[str, pd.DataFrame]:
    return {
        key: normalize_frame(key, read_csv_sheet(cfg["csv"]))
        for key, cfg in SHEET_CONFIG.items()
    }


def parse_clock(value: Any) -> float:
    """Convert a time-like value to minutes since midnight."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, time):
        return value.hour * 60 + value.minute + value.second / 60
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.hour * 60 + value.minute + value.second / 60
    text = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"):
        try:
            parsed = datetime.strptime(text, fmt).time()
            return parsed.hour * 60 + parsed.minute + parsed.second / 60
        except ValueError:
            continue
    return np.nan


def parse_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse dates day-first while retaining invalid/missing values as NaT."""
    if "Date" in frame.columns:
        frame["Date_original"] = frame["Date"].astype("string")
        frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    return frame


def derive_block(hour: pd.Series) -> pd.Series:
    """Assign the configured AM, MD and PM blocks from numeric hours."""
    conditions = [
        hour.lt(11),
        hour.ge(11) & hour.lt(15),
        hour.ge(15),
    ]
    return pd.Series(
        np.select(conditions, ["AM", "MD", "PM"], default=None),
        index=hour.index,
        dtype="string",
    )


def normalize_frame(sheet_key: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized date, time, hour, block, route and source fields."""
    frame = parse_date_column(frame)
    event_time = SHEET_CONFIG[sheet_key]["event_time"]
    time_columns = {
        column
        for column in (event_time, "Time joined", "Time boarded", "Arrived", "Departed")
        if column and column in frame.columns
    }
    for column in time_columns:
        frame[f"{column} minutes"] = frame[column].map(parse_clock).astype("float64")
    if event_time and event_time in frame.columns:
        minutes = frame[f"{event_time} minutes"]
        frame["hour"] = minutes / 60.0
        derived_block = derive_block(frame["hour"])
        frame["Block"] = derived_block.fillna(
            frame.get("Block", pd.Series(pd.NA, index=frame.index, dtype="string"))
            .astype("string")
            .str.upper()
        )
        frame["time_of_day"] = pd.to_datetime(
            minutes, unit="m", origin="1899-12-30", errors="coerce"
        ).dt.time
    else:
        frame["hour"] = np.nan
        frame["time_of_day"] = pd.Series(
            pd.NA, index=frame.index, dtype="object"
        )
    if "Day" in frame.columns:
        frame["Day"] = frame["Day"].astype("string").str.strip().str.title()
    if "Stop" in frame.columns and "Destination / direction" in frame.columns:
        stop = frame["Stop"].astype("string").str.strip()
        destination = frame["Destination / direction"].astype("string").str.strip()
        frame["stop_direction"] = stop.fillna("Unknown stop") + " -> " + destination.fillna(
            "Unknown destination"
        )
    if "Date" in frame.columns and "Day" in frame.columns:
        parsed_day = frame["Date"].dt.day_name()
        frame["date_day_mismatch"] = (
            frame["Date"].notna()
            & frame["Day"].notna()
            & parsed_day.ne(frame["Day"])
        )
    else:
        frame["date_day_mismatch"] = False
    return frame


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or an all-missing series when absent."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def prepare_wait_data(wait: pd.DataFrame) -> pd.DataFrame:
    """Recompute waits from joined/boarded clocks and keep censoring explicit."""
    frame = wait.copy()
    joined = frame.get("Time joined", pd.Series(pd.NA, index=frame.index)).map(
        parse_clock
    )
    boarded = frame.get("Time boarded", pd.Series(pd.NA, index=frame.index)).map(
        parse_clock
    )
    frame["joined_minutes"] = pd.to_numeric(joined, errors="coerce")
    frame["boarded_minutes"] = pd.to_numeric(boarded, errors="coerce")
    frame["end_before_start"] = (
        frame["boarded_minutes"].notna()
        & frame["joined_minutes"].notna()
        & frame["boarded_minutes"].lt(frame["joined_minutes"])
    )
    unfinished = pd.Series(False, index=frame.index)
    if "Unfinished?" in frame.columns:
        unfinished = (
            frame["Unfinished?"]
            .astype("string")
            .str.strip()
            .str.casefold()
            .isin({"yes", "true", "1"})
        )
    frame["censored"] = frame["boarded_minutes"].isna() | unfinished
    frame["wait_minutes"] = frame["boarded_minutes"] - frame["joined_minutes"]
    invalid = frame["end_before_start"] | frame["joined_minutes"].isna()
    frame.loc[frame["censored"] | invalid, "wait_minutes"] = np.nan
    frame.loc[frame["wait_minutes"].lt(0), "wait_minutes"] = np.nan
    frame["wait_event"] = frame["wait_minutes"].notna()
    frame["censor_time_available"] = False
    if "Type used" in frame.columns:
        normalized_type = (
            frame["Type used"].astype("string").str.strip().str.casefold()
        )
        frame["Type used"] = normalized_type.map(
            {"cng bus": "CNG bus", "keke": "Keke"}
        ).fillna(frame["Type used"])
    return frame


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Compute a Wilson 95% confidence interval for a proportion."""
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def vehicle_efficiency(
    queue: pd.DataFrame, vehicle: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute headway, route load/left-behind, and demand-capacity metrics."""
    vehicles = vehicle.copy()
    vehicles["arrived_minutes"] = vehicles.get(
        "Arrived minutes", pd.Series(np.nan, index=vehicles.index)
    )
    vehicles["capacity_numeric"] = _numeric(vehicles, "Capacity")
    if "Type" in vehicles.columns:
        type_capacity = (
            vehicles["Type"].astype("string").str.strip().str.casefold().map(
                {"cng bus": CAPACITIES["CNG bus"], "keke": CAPACITIES["Keke"]}
            )
        )
        vehicles["capacity_numeric"] = vehicles["capacity_numeric"].fillna(
            type_capacity
        )
    vehicles["boarded_numeric"] = _numeric(vehicles, "Boarded")
    vehicles["left_behind_numeric"] = _numeric(vehicles, "Left behind")
    vehicles["load_factor"] = (
        vehicles["boarded_numeric"] / vehicles["capacity_numeric"]
    )
    vehicles["hour_slot"] = np.floor(vehicles["hour"])
    vehicles["headway_min"] = vehicles.sort_values(
        ["Date", "stop_direction", "arrived_minutes"]
    ).groupby(["Date", "stop_direction"], dropna=False)["arrived_minutes"].diff()
    vehicles.loc[vehicles["headway_min"].le(0), "headway_min"] = np.nan

    headway = (
        vehicles.groupby(["stop_direction", "hour_slot"], dropna=False)
        .agg(
            mean_headway_min=("headway_min", "mean"),
            median_headway_min=("headway_min", "median"),
            n=("headway_min", "count"),
        )
        .reset_index()
    )
    load = (
        vehicles.groupby(["stop_direction", "Type"], dropna=False)
        .agg(
            mean_load_factor=("load_factor", "mean"),
            boarded_total=("boarded_numeric", "sum"),
            capacity_total=("capacity_numeric", "sum"),
            left_behind_total=("left_behind_numeric", "sum"),
            n=("Vehicle ID", "count"),
        )
        .reset_index()
    )
    load["left_behind_rate"] = load["left_behind_total"] / (
        load["boarded_total"] + load["left_behind_total"]
    )
    load["load_factor_pct"] = load["mean_load_factor"] * 100

    queue_hour = queue.copy()
    queue_hour["hour_slot"] = np.floor(queue_hour["hour"])
    queue_hour["arrivals_numeric"] = _numeric(queue_hour, "New arrivals (10 min)")
    arrivals = (
        queue_hour.groupby(["Date", "stop_direction", "hour_slot"], dropna=False)
        .agg(arrivals_per_hour=("arrivals_numeric", "sum"))
        .reset_index()
        .groupby(["stop_direction", "hour_slot"], dropna=False)
        .agg(
            demand_per_hour=("arrivals_per_hour", "mean"),
            demand_n=("arrivals_per_hour", "count"),
        )
        .reset_index()
    )
    departed_minutes = vehicles.get(
        "Departed minutes", pd.Series(np.nan, index=vehicles.index)
    )
    vehicles["departed_hour_slot"] = np.floor(
        pd.to_numeric(departed_minutes, errors="coerce") / 60
    )
    vehicles["departed_capacity"] = vehicles["capacity_numeric"].where(
        pd.to_numeric(departed_minutes, errors="coerce").notna(), np.nan
    )
    capacity = (
        vehicles.groupby(
            ["Date", "stop_direction", "departed_hour_slot"], dropna=False
        )
        .agg(capacity_per_hour=("departed_capacity", "sum"))
        .reset_index()
        .rename(columns={"departed_hour_slot": "hour_slot"})
        .groupby(["stop_direction", "hour_slot"], dropna=False)
        .agg(
            seats_leaving_per_hour=("capacity_per_hour", "mean"),
            capacity_n=("capacity_per_hour", "count"),
        )
        .reset_index()
    )
    demand_capacity = arrivals.merge(
        capacity, on=["stop_direction", "hour_slot"], how="outer"
    )
    demand_capacity["demand_to_capacity_ratio"] = (
        demand_capacity["demand_per_hour"]
        / demand_capacity["seats_leaving_per_hour"].replace(0, np.nan)
    )
    demand_capacity["n"] = demand_capacity[["demand_n", "capacity_n"]].max(axis=1)
    return headway, load, demand_capacity





def load_frames() -> dict[str, pd.DataFrame]:
    """Load all observation sheets and add the derived fields the app uses."""
    frames = load_all_data()
    wait = prepare_wait_data(frames["wait"])
    for key in ("queue", "vehicle"):
        frames[key]["hour_slot"] = np.floor(frames[key]["hour"])
    wait["hour_slot"] = np.floor(wait["hour"])
    frames["wait"] = wait
    frames["vehicle"]["Type"] = frames["vehicle"]["Type"].astype("string").str.strip()
    for key in ("queue", "wait", "vehicle"):
        f = frames[key]
        f["Day"] = f["Day"].fillna(f["Date"].dt.day_name())
        f["is_weekend"] = f["Day"].isin(["Saturday", "Sunday"])
        for col in ("stop_direction", "Block", "Day", "Stop", "Type", "Type used"):
            if col in f:
                f[col] = f[col].astype(object).where(f[col].notna(), None)
    # The 12 stop-directions on the observation rota (defined by the queue sheet);
    # a handful of stray rows (missing or same-stop destinations) are excluded.
    main_routes = set(frames["queue"]["stop_direction"].dropna()) - {
        r for r in frames["queue"]["stop_direction"].dropna() if "Unknown" in r
    }
    for key in ("queue", "wait", "vehicle"):
        frames[key]["valid_route"] = frames[key]["stop_direction"].isin(main_routes)
        # Field rows carry the recorder's name in "Entered by"; model-estimated rows
        # carry only a "Roster pair (reference)". The two never overlap in the workbook.
        frames[key]["source"] = np.where(frames[key]["Entered by"].notna(), "Field", "Estimated")
    return frames


# ---------------------------------------------------------------------------
# Sheets the field team left empty, filled from data that WAS collected
# ---------------------------------------------------------------------------
MAIN_WEEK_START = pd.Timestamp("2026-10-26")


def fleet_availability(vehicle: pd.DataFrame) -> pd.DataFrame:
    """Sheet D rebuilt from sheet C: one row per vehicle per day of the main week.

    A vehicle counts as 'Operating' on a day if it was logged arriving at any
    observed stop that day. Vehicles in the week's fleet list that were never
    logged that day are 'Not observed' (they may have been parked, faulty or
    working routes we did not watch; the field sheet cannot tell which).
    """
    v = vehicle[vehicle["valid_route"] & vehicle["Date"].ge(MAIN_WEEK_START)].dropna(subset=["Vehicle ID"])
    fleet = v.groupby("Vehicle ID")["Type"].agg(lambda s: s.mode().iat[0]).reset_index()
    days = sorted(v["Date"].dropna().unique())
    grid = fleet.merge(pd.DataFrame({"Date": days}), how="cross")
    seen = (v.groupby(["Date", "Vehicle ID"])
            .agg(arrivals=("Arrived", "size"),
                 first_seen=("Arrived minutes", "min"), last_seen=("Arrived minutes", "max"),
                 stops=("Stop", lambda s: ", ".join(sorted(s.dropna().unique()))))
            .reset_index())
    out = grid.merge(seen, on=["Date", "Vehicle ID"], how="left")
    out["Status"] = np.where(out["arrivals"].notna(), "Operating", "Not observed")
    out["arrivals"] = out["arrivals"].fillna(0).astype(int)
    fmt = lambda m: f"{int(m // 60):02d}:{int(m % 60):02d}" if pd.notna(m) else ""
    out["first_seen"] = out["first_seen"].map(fmt)
    out["last_seen"] = out["last_seen"].map(fmt)
    out["Day"] = pd.to_datetime(out["Date"]).dt.day_name()
    out = out.rename(columns={"arrivals": "Arrivals logged", "first_seen": "First seen",
                              "last_seen": "Last seen", "stops": "Stops served"})
    out["Source"] = "Derived from sheet C (vehicle & trip log)"
    return out[["Date", "Day", "Vehicle ID", "Type", "Status", "Arrivals logged",
                "First seen", "Last seen", "Stops served", "Source"]].sort_values(["Date", "Type", "Vehicle ID"])


def holdout_validation(wait: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-day-out test of a simple wait predictor.

    For each day of the main week, the model is built from the OTHER six days:
    predicted wait = median wait for the same stop-direction, time block and
    vehicle type; the 80% range = 10th–90th percentile; the chance of boarding
    within 10 minutes = share of those training passengers who did. It is then
    scored against every passenger actually timed on the held-out day. The
    baseline predicts the overall median wait for everyone.
    """
    w = wait[wait["valid_route"] & wait["wait_event"] & wait["Date"].ge(MAIN_WEEK_START)].copy()
    key = ["stop_direction", "Block", "Type used"]
    rows = []
    for day in sorted(w["Date"].unique()):
        train, test = w[w["Date"] != day], w[w["Date"] == day]
        g = (train.groupby(key)["wait_minutes"]
             .agg(pred="median", lo=lambda s: s.quantile(.1), hi=lambda s: s.quantile(.9),
                  p10=lambda s: s.le(10).mean(), n_train="count").reset_index())
        t = test.merge(g, on=key, how="inner")
        t["baseline"] = train["wait_minutes"].median()
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["abs_error"] = (t["wait_minutes"] - t["pred"]).abs()
    t["baseline_abs_error"] = (t["wait_minutes"] - t["baseline"]).abs()
    t["within_80"] = t["wait_minutes"].between(t["lo"], t["hi"])
    t["boarded_10"] = t["wait_minutes"].le(10)
    return t


def validation_sheet(t: pd.DataFrame) -> pd.DataFrame:
    """Hold-out results laid out like the team's Validation template."""
    clock = lambda m: f"{int(m // 60):02d}:{int(m % 60):02d}" if pd.notna(m) else ""
    return pd.DataFrame({
        "Date": t["Date"].dt.strftime("%d/%m/%Y"), "Day": t["Day"], "Block": t["Block"], "Stop": t["Stop"],
        "Observer pair": t["Roster pair (reference)"], "Destination / direction": t["Destination / direction"],
        "Type used": t["Type used"], "Time joined": t["joined_minutes"].map(clock),
        "Predicted wait (min)": t["pred"].round(1), "80% low (min)": t["lo"].round(1), "80% high (min)": t["hi"].round(1),
        "Predicted chance of boarding within 10 min": t["p10"].round(3),
        "Time boarded": t["boarded_minutes"].map(clock), "Observed wait (min)": t["wait_minutes"],
        "Within 80% range?": np.where(t["within_80"], "Yes", "No"), "Absolute error (min)": t["abs_error"].round(1),
        "Boarded within 10 min?": np.where(t["boarded_10"], "Yes", "No"),
        "Method": "Leave-one-day-out: model built from the other 6 days of the main week",
    })


# ==========================================================================
# SURVIVAL ANALYSIS AND QUEUE SIMULATION (formerly cng_sim.py)
# ==========================================================================

import numpy as np
import pandas as pd

END_OF_DAY = 18 * 60.0
PCTS = (50, 75, 90, 95)


# ---------------------------------------------------------------------------
# Survival analysis (Kaplan–Meier) with unfinished waits censored at 18:00
# ---------------------------------------------------------------------------
def survival_input(wait: pd.DataFrame) -> pd.DataFrame:
    """Durations and event flags: completed waits are events; unfinished ones are
    censored at the end of observation (18:00) — their wait was at least that long."""
    w = wait.copy()
    dur = w["wait_minutes"].copy()
    cens = w["censored"] & w["joined_minutes"].notna()
    dur[cens] = (END_OF_DAY - w.loc[cens, "joined_minutes"]).clip(lower=0)
    w["duration"] = dur
    w["event"] = ~w["censored"]
    return w.dropna(subset=["duration"])


def kaplan_meier(duration: pd.Series, event: pd.Series) -> pd.DataFrame:
    """Return the share still waiting S(t) at each minute t."""
    d = pd.DataFrame({"t": np.round(duration.values.astype(float)), "e": event.values.astype(bool)})
    times = np.arange(0, int(np.ceil(d["t"].max())) + 1)
    n_at_risk = len(d)
    s = 1.0
    out = []
    grouped = d.groupby("t")
    events = grouped["e"].sum()
    totals = grouped.size()
    for t in times:
        out.append((t, s))  # S just before leaving time t is S(t-)
        if t in totals.index:
            e = events.get(t, 0)
            if n_at_risk > 0:
                s *= 1 - e / n_at_risk
            n_at_risk -= totals[t]
    surv = pd.DataFrame(out, columns=["t", "still_waiting"])
    # S(t) after events at t
    surv["still_waiting"] = surv["still_waiting"].shift(-1).fillna(s)
    return surv


def km_percentile(surv: pd.DataFrame, p: float) -> float:
    """Smallest t with S(t) <= 1 - p/100 (the p-th percentile wait)."""
    hit = surv.loc[surv["still_waiting"] <= 1 - p / 100, "t"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def area_under_curve(surv: pd.DataFrame) -> float:
    """Area under S(t) = (restricted) mean wait in minutes."""
    return float(surv["still_waiting"].sum())


def percentile_table(wait: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """P50/P75/P90/P95 and area under the curve, KM-adjusted for unfinished waits."""
    w = survival_input(wait)
    groups = [((), w)] if not by else w.groupby(by)
    rows = []
    for key, g in groups:
        surv = kaplan_meier(g["duration"], g["event"])
        row = dict(zip(by or [], key if isinstance(key, tuple) else (key,)))
        row.update({f"P{p}": km_percentile(surv, p) for p in PCTS})
        row["area"] = area_under_curve(surv)
        row["n"] = len(g)
        row["unfinished"] = int((~g["event"]).sum())
        row["over15"] = float(surv.loc[surv["t"] == 15, "still_waiting"].iat[0]) if (surv["t"] == 15).any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def tail_share(waits: pd.Series, p: float = 90) -> float:
    """Share of all passenger-minutes spent by the passengers above the p-th percentile."""
    w = waits.dropna()
    cut = np.percentile(w, p)
    return float(w[w > cut].sum() / w.sum()) if w.sum() else np.nan


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _arrivals(qd: pd.DataFrame) -> np.ndarray:
    """Passenger arrival times (minutes) for one route-day from sheet B1.

    Each count is "new arrivals in the last 10 minutes". When two counts were taken
    less than 10 minutes apart the windows overlap, so only the new part of the
    window (gap / 10 of the count) is added, to avoid counting people twice."""
    qd = qd.dropna(subset=["Time minutes"]).sort_values("Time minutes")
    times: list[float] = []
    prev = None
    for t, new, queue in zip(qd["Time minutes"], qd["New arrivals (10 min)"], qd["Queue size"]):
        if prev is None:  # people already queuing when observation began
            times += [t] * int(max(queue, 0))
            prev = t
            continue
        gap = t - prev
        if gap <= 0:
            continue
        span = min(gap, 10.0)
        n = int(round(max(new, 0) * span / 10.0))
        if n:
            times += list(t - span + (np.arange(n) + 0.5) * span / n)
        prev = t
    return np.sort(np.array(times, dtype=float))


def _vehicles(vd: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vd = vd.dropna(subset=["Arrived minutes"]).sort_values("Arrived minutes")
    cap = pd.to_numeric(vd["Spaces free on arrival"], errors="coerce")
    full_cap = pd.to_numeric(vd["Capacity"], errors="coerce").fillna(vd["Type"].map({"CNG bus": 13, "Keke": 5}))
    cap = cap.fillna(full_cap)
    arr = vd["Arrived minutes"].to_numpy(float)
    dep = pd.to_numeric(vd["Departed minutes"], errors="coerce").to_numpy(float)
    dep = np.where(np.isnan(dep) | (dep < arr), arr, dep)
    return arr, dep, cap.to_numpy(float), vd["Type"].astype(str).to_numpy()


def serve(arr: np.ndarray, veh_t: np.ndarray, veh_cap: np.ndarray, veh_dep: np.ndarray | None = None) -> np.ndarray:
    """FIFO boarding. A vehicle loads everyone already queuing when it arrives and
    anyone who walks up while it is still at the stop (until it departs), up to its
    free seats. Returns each passenger's wait (NaN = still waiting at 18:00)."""
    if veh_dep is None:
        veh_dep = veh_t
    order = np.argsort(veh_t)
    veh_t, veh_cap, veh_dep = veh_t[order], veh_cap[order], veh_dep[order]
    waits = np.full(len(arr), np.nan)
    head = 0
    for t, c, d in zip(veh_t, veh_cap, veh_dep):
        avail_end = np.searchsorted(arr, max(d, t), side="right")
        take = int(min(c, max(avail_end - head, 0)))
        if take:
            waits[head:head + take] = np.maximum(t - arr[head:head + take], 0)
            head += take
    return waits


def simulate(queue: pd.DataFrame, vehicle: pd.DataFrame, modify=None) -> pd.DataFrame:
    """Simulate every route-day. `modify(route, date, veh_t, veh_cap, veh_type)` may
    return altered (veh_t, veh_cap) to test a scenario."""
    out = []
    vgroups = {k: g for k, g in vehicle.groupby(["Date", "stop_direction"])}
    for (date, route), qd in queue.groupby(["Date", "stop_direction"]):
        arr = _arrivals(qd)
        if not len(arr):
            continue
        vd = vgroups.get((date, route))
        if vd is None:
            vt, vdep, vc, vty = (np.array([]),) * 4
        else:
            vt, vdep, vc, vty = _vehicles(vd)
        dwell = np.median(vdep - vt) if len(vt) else 0.0
        if modify is not None:
            n0 = len(vt)
            keep_dwell = dict(zip(vt, vdep - vt))
            vt, vc = modify(route, date, vt, vc, vty)
            # moved vehicles keep their dwell time; added ones get the route's median dwell
            vdep = np.array([t + keep_dwell.get(t, dwell) for t in vt])
        wt = serve(arr, vt, vc, vdep)
        out.append(pd.DataFrame({"Date": date, "stop_direction": route, "joined_minutes": arr, "sim_wait": wt}))
    sim = pd.concat(out, ignore_index=True)
    sim["hour_slot"] = np.floor(sim["joined_minutes"] / 60)
    sim["Block"] = np.select([sim["hour_slot"] < 11, sim["hour_slot"] < 15], ["AM", "MD"], "PM")
    sim["censored"] = sim["sim_wait"].isna()
    sim["wait_minutes"] = sim["sim_wait"]
    sim["wait_event"] = ~sim["censored"]
    return sim


def summarize_sim(sim: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    return percentile_table(sim, by)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_even_spacing(route, date, vt, vc, vty):
    """Same vehicles each hour, but arriving at evenly spaced times (no bunching)."""
    if not len(vt):
        return vt, vc
    hours = np.floor(vt / 60)
    new_t = vt.copy()
    for h in np.unique(hours):
        m = hours == h
        k = m.sum()
        new_t[m] = h * 60 + (np.arange(k) + 0.5) * 60 / k
    return new_t, vc


def scenario_add_peak(routes_hours: set[tuple[str, int]], extra_per_hour: int = 2, seats: int = 13):
    """Add `extra_per_hour` CNG buses, evenly spaced, in the given (route, hour) cells."""
    def modify(route, date, vt, vc, vty):
        add_t = [h * 60 + (i + 0.5) * 60 / extra_per_hour
                 for (r, h) in routes_hours if r == route for i in range(extra_per_hour)]
        if not add_t:
            return vt, vc
        return np.concatenate([vt, add_t]), np.concatenate([vc, [seats] * len(add_t)])
    return modify


def scenario_reallocate(targets: set[tuple[str, int]], donors: list[str], per_hour: int = 2):
    """Take `per_hour` CNG buses from donor routes in the same hour and run them on target cells."""
    def modify(route, date, vt, vc, vty):
        hours = np.floor(vt / 60) if len(vt) else np.array([])
        if route in donors:
            target_hours = {h for (_, h) in targets}
            keep = np.ones(len(vt), bool)
            for h in target_hours:
                idx = np.where((hours == h) & (vty == "CNG bus"))[0]
                # each target cell needs per_hour buses; spread the donation over donors
                n_take = int(np.ceil(per_hour * sum(1 for (_, hh) in targets if hh == h) / len(donors)))
                keep[idx[:n_take]] = False
            return vt[keep], vc[keep]
        add_t = [h * 60 + (i + 0.5) * 60 / per_hour for (r, h) in targets if r == route for i in range(per_hour)]
        if add_t:
            return np.concatenate([vt, add_t]), np.concatenate([vc, [13] * len(add_t)])
        return vt, vc
    return modify


def compose(*mods):
    """Apply several scenario modifiers in turn."""
    def modify(route, date, vt, vc, vty):
        types = vty
        for m in mods:
            n_before = len(vt)
            vt, vc = m(route, date, vt, vc, types)
            if len(vt) != n_before:  # vehicles were added/removed: treat added ones as buses
                types = np.array(list(types[:min(len(types), len(vt))]) + ["CNG bus"] * max(0, len(vt) - len(types)))
        return vt, vc
    return modify


HOT_CELLS = (
    {("NEW MARKET -> ACADEMICS", h) for h in (8, 9, 10)}
    | {("CAMPUS GATE -> BUS STOP", h) for h in (8, 9, 10)}
    | {("BUS STOP -> CAMPUS GATE", h) for h in (15, 16, 17)}
)
DONORS = ["CAMPUS GATE -> HALL", "HALL -> CAMPUS GATE"]


def scenario_table(queue: pd.DataFrame, vehicle: pd.DataFrame, extra: int = 2) -> tuple[pd.DataFrame, dict]:
    """Run the standard set of what-if scenarios. Returns a summary table and the simulations."""
    days = queue["Date"].nunique()
    runs = {
        "As observed": None,
        "Even spacing (same vehicles)": scenario_even_spacing,
        f"Move {extra} buses/hour from Hall ↔ Campus Gate to the 9 peak route-hours": scenario_reallocate(HOT_CELLS, DONORS, extra),
        f"Add {extra} buses/hour in the 9 peak route-hours": scenario_add_peak(HOT_CELLS, extra),
        f"Move {extra} buses/hour + even spacing": compose(scenario_reallocate(HOT_CELLS, DONORS, extra), scenario_even_spacing),
    }
    sims, rows = {}, []
    for label, mod in runs.items():
        sim = simulate(queue, vehicle, mod)
        sims[label] = sim
        o = percentile_table(sim).iloc[0]
        c = sim[sim["wait_event"]]
        donor = percentile_table(sim[sim["stop_direction"].isin(DONORS)]).iloc[0]
        hot = sim.set_index(["stop_direction", "hour_slot"]).index.isin(list(HOT_CELLS))
        hot_t = percentile_table(sim[hot]).iloc[0]
        rows.append({"scenario": label, "P50": o["P50"], "P75": o["P75"], "P90": o["P90"], "P95": o["P95"],
                     "area": o["area"], "over15": o["over15"],
                     "hours_per_day": c["wait_minutes"].sum() / 60 / days,
                     "peak_P90": hot_t["P90"], "donor_P90": donor["P90"], "donor_area": donor["area"]})
    t = pd.DataFrame(rows)
    base = t.iloc[0]
    t["area_change"] = t["area"] / base["area"] - 1
    t["hours_saved_per_day"] = base["hours_per_day"] - t["hours_per_day"]
    return t, sims


# Short names used below: D.<data helper>, S.<simulation helper>
D = S = types.SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith('__')})

HERE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="OAU Campus Transport Study",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="auto",
)

# ==========================================================================
# Design system
# ==========================================================================
INK = "#1d2a2e"
MUTED = "#5b6b70"
TEAL = "#1f6f78"
AMBER = "#d99a00"
RED = "#c8553d"
PAPER = "#fbfaf7"
TYPE_COLORS = {"CNG bus": TEAL, "Keke": AMBER}
BLOCK_COLORS = {"AM": "#3b7dd8", "MD": "#8a63d2", "PM": RED}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FONT = "Inter, -apple-system, Segoe UI, sans-serif"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea, button {{
    font-family: {FONT};
}}
.block-container {{ padding-top: 4.2rem; padding-bottom: 1rem; max-width: 1240px; }}
h1, h2, h3, .serif {{ font-family: 'Source Serif 4', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700 !important; }}
h2 {{ font-weight: 600 !important; margin-top: 2.4rem !important; padding-top: .6rem;
      border-top: 1px solid rgba(0,0,0,.08); }}
h3 {{ font-weight: 600 !important; margin-top: 1.4rem !important; }}
.article p, .article li {{ font-size: 1.04rem; line-height: 1.7; color: #26363b; }}
.kicker {{ text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; font-weight: 600; color: {TEAL}; }}
.dek {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.28rem; line-height: 1.55; color: #34474d; max-width: 60rem; }}
.byline {{ color: {MUTED}; font-size: .92rem; margin: .6rem 0 1.4rem; }}
div[data-testid="stMetric"] {{
    background: white; border: 1px solid rgba(31,111,120,.16); border-radius: 12px;
    padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.03);
}}
div[data-testid="stMetricLabel"] p {{ font-size: .82rem; color: {MUTED}; }}
div[data-testid="stMetricValue"] {{ font-family: 'Source Serif 4', Georgia, serif; color: {INK}; font-size: 1.85rem; }}
.explain {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin: -.4rem 0 1.6rem;
    border: 1px solid rgba(31,111,120,.18); border-radius: 12px; overflow: hidden; background: white; }}
.explain > div {{ padding: .85rem 1.05rem; font-size: .93rem; line-height: 1.55; color: #2c3c41; }}
.explain > div:first-child {{ background: rgba(31,111,120,.05); border-right: 1px solid rgba(31,111,120,.14); }}
.explain b.lbl {{ display: block; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: {TEAL}; margin-bottom: .25rem; }}
@media (max-width: 760px) {{ .explain {{ grid-template-columns: 1fr; }}
    .explain > div:first-child {{ border-right: none; border-bottom: 1px solid rgba(31,111,120,.14); }} }}
.callout {{ border-left: 4px solid {TEAL}; background: rgba(31,111,120,.06); padding: .9rem 1.1rem;
    border-radius: 0 10px 10px 0; margin: 1rem 0 1.4rem; font-size: 1rem; line-height: 1.6; }}
.callout.warn {{ border-left-color: {AMBER}; background: rgba(217,154,0,.08); }}
.pipeline {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: stretch; margin: .6rem 0 1.4rem; }}
.pipeline .step {{ flex: 1 1 150px; background: white; border: 1px solid rgba(0,0,0,.08); border-radius: 12px; padding: .75rem .9rem; }}
.pipeline .step b {{ display:block; color: {TEAL}; font-size: .78rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .2rem; }}
.pipeline .step span {{ font-size: .9rem; color: #33444a; line-height: 1.45; }}
.pipeline .arrow {{ align-self: center; color: {MUTED}; font-size: 1.3rem; }}
.sheet-card {{ background: white; border: 1px solid rgba(0,0,0,.08); border-radius: 12px; padding: 1rem 1.1rem; height: 100%; }}
.sheet-card h4 {{ margin: 0 0 .35rem; font-family: 'Source Serif 4', Georgia, serif; font-size: 1.08rem; color: {INK}; }}
.sheet-card .meta {{ font-size: .8rem; color: {TEAL}; font-weight: 600; margin-bottom: .4rem; }}
.sheet-card p {{ font-size: .92rem; line-height: 1.55; margin: 0; color: #33444a; }}
.team-card {{ background: white; border: 1px solid rgba(0,0,0,.08); border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: .9rem; min-height: 150px; }}
.team-card .avatar {{ width: 42px; height: 42px; border-radius: 50%; background: {TEAL}; color: white;
    display: inline-flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: .5rem; }}
.team-card .nm {{ font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.1rem; color: {INK}; }}
.team-card .rl {{ font-size: .8rem; color: {TEAL}; font-weight: 600; margin-bottom: .35rem; }}
.team-card .ct {{ font-size: .9rem; color: #33444a; line-height: 1.5; }}
.site-footer {{ margin-top: 3rem; padding: 1.4rem 0 .4rem; border-top: 1px solid rgba(0,0,0,.1); color: {MUTED}; font-size: .86rem; line-height: 1.6; }}
.site-footer b {{ color: {INK}; }}
table.metrics {{ width: 100%; border-collapse: collapse; font-size: .93rem; margin: .4rem 0 1.4rem; }}
table.metrics th {{ text-align: left; font-size: .75rem; letter-spacing: .1em; text-transform: uppercase; color: {TEAL}; border-bottom: 2px solid rgba(31,111,120,.3); padding: .5rem .6rem; }}
table.metrics td {{ border-bottom: 1px solid rgba(0,0,0,.07); padding: .6rem; vertical-align: top; line-height: 1.5; color: #2c3c41; }}
table.metrics td:first-child {{ font-weight: 600; color: {INK}; white-space: nowrap; }}
.takeaway {{ border: 1px solid rgba(31,111,120,.25); background: linear-gradient(90deg, rgba(31,111,120,.10), rgba(31,111,120,.02));
    border-radius: 12px; padding: .8rem 1.1rem; margin: .2rem 0 1.2rem; font-size: 1.02rem; line-height: 1.55; color: {INK}; }}
.takeaway b.lbl {{ display:block; font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:{TEAL}; margin-bottom:.2rem; }}
.source-tag {{ display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
    padding:.15rem .55rem; border-radius:999px; margin-right:.4rem; vertical-align:middle; }}
.source-tag.field {{ background:#1f6f78; color:white; }}
.source-tag.model {{ background:#f3e2b5; color:#6b4e00; }}
.anchor {{ position: relative; top: -70px; }}
.toc a {{ display:block; padding:.22rem 0; color:{INK}; text-decoration:none; font-size:.92rem; }}
.toc a:hover {{ color:{TEAL}; }}
.toc a.sub {{ padding-left: .9rem; font-size:.84rem; color:{MUTED}; }}
[data-testid="stAppDeployButton"], .stAppDeployButton {{ display:none !important; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
[data-testid="stMetricLabel"] p {{ line-height: 1.25 !important; }}
div[data-testid="stPlotlyChart"] {{ border-radius: 12px; }}
@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; padding-top: 3.6rem !important; }}
  h1 {{ font-size: 1.8rem !important; line-height: 1.2 !important; }}
  h2 {{ font-size: 1.4rem !important; }}
  h3 {{ font-size: 1.15rem !important; }}
  .dek {{ font-size: 1.04rem !important; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.45rem !important; }}
  div[data-testid="stMetric"] {{ padding: 8px 10px !important; }}
  table.metrics {{ display:block; overflow-x:auto; font-size:.85rem; }}
  .pipeline .arrow {{ display:none; }}
  .article p, .article li {{ font-size: .98rem; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


def style(fig: go.Figure, height: int = 420, title: str | None = None,
          subtitle: str | None = None) -> go.Figure:
    text = f"<b>{title}</b>" if title else None
    if title and subtitle:
        text += f"<br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=13, color=INK),
        margin=dict(l=10, r=20, t=100 if subtitle else 80, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT),
    )
    if text:
        fig.update_layout(title=dict(text=text, x=0, y=0.97, yanchor="top", yref="container",
                                     font=dict(size=17, family="Source Serif 4, Georgia, serif")))
    fig.update_xaxes(gridcolor="rgba(0,0,0,.06)", zeroline=False, fixedrange=True, automargin=True)
    fig.update_yaxes(gridcolor="rgba(0,0,0,.06)", zeroline=False, fixedrange=True, automargin=True)
    fig.update_layout(dragmode=False)
    return fig


def show(fig: go.Figure) -> None:
    # No toolbar and no zoom: charts stay readable when scrolled past on a phone
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False,
                                                  "doubleClick": False, "responsive": True})


def takeaway(text: str) -> None:
    st.markdown(f"<div class='takeaway'><b class='lbl'>Key takeaway</b>{text}</div>", unsafe_allow_html=True)


def section(title: str, anchor: str, level: int = 2) -> None:
    st.markdown(f"<div id='{anchor}' class='anchor'></div>", unsafe_allow_html=True)
    st.markdown(("## " if level == 2 else "### ") + title)


def explain(how: str, what: str) -> None:
    """Plain-language 'how to read' / 'what it shows' panel under a chart."""
    st.markdown(
        f"<div class='explain'><div><b class='lbl'>How to read this chart</b>{how}</div>"
        f"<div><b class='lbl'>What it shows</b>{what}</div></div>",
        unsafe_allow_html=True,
    )


def callout(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='callout{' warn' if warn else ''}'>{text}</div>", unsafe_allow_html=True)


def md(text: str) -> None:
    st.markdown(f"<div class='article'>\n\n{text}\n\n</div>", unsafe_allow_html=True)


def hh(h: float) -> str:
    return f"{int(h):02d}:00" if pd.notna(h) else "—"


# ==========================================================================
# Data
# ==========================================================================
@st.cache_data(show_spinner="Loading field observations…")
def frames() -> dict[str, pd.DataFrame]:
    return D.load_frames()


REQUIRED = ["B1_Queue_and_Demand.csv", "B2_Wait_Sample.csv", "C_Vehicle_and_Trip.csv"]
_missing = []
for _name in REQUIRED:
    try:
        find_file(_name)
    except FileNotFoundError:
        _missing.append(_name)
if _missing:
    _here = sorted(f.name for f in Path(__file__).resolve().parent.iterdir() if f.is_file())
    st.error("**Some data files are missing from the GitHub repository:** " + ", ".join(f"`{m}`" for m in _missing)
             + ".\n\nUpload them next to `app.py` (same folder, no subfolders). Files found right now: "
             + ", ".join(f"`{f}`" for f in _here))
    st.stop()


@st.cache_data
def table(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(find_file(name))
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def team() -> pd.DataFrame:
    try:
        return pd.read_csv(find_file("team.csv")).fillna("")
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "role", "contributions"])


F = frames()
# Model-estimated main week (26 Oct – 1 Nov): used for the detailed patterns
Q_ALL = F["queue"][F["queue"]["valid_route"] & F["queue"]["source"].eq("Estimated")]
W_ALL = F["wait"][F["wait"]["valid_route"] & F["wait"]["source"].eq("Estimated")]
V_ALL = F["vehicle"][F["vehicle"]["valid_route"] & F["vehicle"]["source"].eq("Estimated")]
# Field records: entered by a named team member (mostly 27–30 September)
_known = lambda f: f["stop_direction"].notna() & ~f["stop_direction"].astype(str).str.contains("Unknown")
Q_F = F["queue"][F["queue"]["source"].eq("Field") & _known(F["queue"])]
W_F = F["wait"][F["wait"]["source"].eq("Field") & _known(F["wait"])]
V_F = F["vehicle"][F["vehicle"]["source"].eq("Field") & _known(F["vehicle"])]
LAST_HOUR = 17  # 18:00 is a single closing snapshot per route-day, so hourly charts stop at 17:00
SURVEY = table("survey_clean.csv")
TESTS = table("hypothesis_tests.csv")
SPEARMAN = table("spearman_correlations.csv")
COMPARE = table("survey_observed_wait_comparison.csv")
TEAM = team()


@st.cache_data(show_spinner=False)
def fleet() -> pd.DataFrame:
    return D.fleet_availability(F["vehicle"][F["vehicle"]["source"].eq("Estimated")])


@st.cache_data(show_spinner=False)
def validation() -> pd.DataFrame:
    return D.holdout_validation(F["wait"][F["wait"]["source"].eq("Estimated")])


FLEET = fleet()
VAL = validation()


def completed(w: pd.DataFrame) -> pd.DataFrame:
    return w.loc[w["wait_event"]]


def vehicle_metrics(v: pd.DataFrame) -> pd.DataFrame:
    v = v.copy()
    v["cap"] = pd.to_numeric(v["Capacity"], errors="coerce").fillna(v["Type"].map(D.CAPACITIES))
    v["boarded_n"] = pd.to_numeric(v["Boarded"], errors="coerce")
    v["left_n"] = pd.to_numeric(v["Left behind"], errors="coerce")
    v["load"] = v["boarded_n"] / v["cap"]
    return v


# ==========================================================================
# Chart builders (shared by the report and the dashboard)
# ==========================================================================
def fig_wait_hist(c: pd.DataFrame) -> go.Figure:
    fig = px.histogram(c, x="wait_minutes", color="Type used", nbins=45, barmode="overlay",
                       opacity=0.72, color_discrete_map=TYPE_COLORS,
                       labels={"wait_minutes": "minutes waited", "Type used": "Type"})
    fig.update_yaxes(title="number of passengers")
    return style(fig, 400, "How long did people wait?", "Completed waits, by vehicle type boarded")


def fig_ecdf(c: pd.DataFrame) -> go.Figure:
    fig = px.ecdf(c, x="wait_minutes", color="Type used", color_discrete_map=TYPE_COLORS,
                  labels={"wait_minutes": "minutes waited", "Type used": "Type"})
    for t in (5, 10, 15):
        fig.add_vline(x=t, line_dash="dot", line_color="#999",
                      annotation_text=f"{t} min", annotation_position="bottom right")
    fig.update_yaxes(title="share already boarded", tickformat=".0%")
    return style(fig, 400, "Share of passengers boarded by each minute",
                 "Read up from a minute mark to see what share had boarded by then")


def fig_wait_by_route(c: pd.DataFrame) -> go.Figure:
    g = (c.groupby("stop_direction")["wait_minutes"]
         .agg(mean="mean", median="median", p90=lambda s: s.quantile(.9), n="count")
         .reset_index().sort_values("mean"))
    fig = go.Figure()
    fig.add_bar(y=g["stop_direction"], x=g["mean"], orientation="h", name="Average wait",
                marker_color=TEAL, customdata=g[["median", "p90", "n"]],
                hovertemplate="%{y}<br>Average %{x:.1f} min · typical (median) %{customdata[0]:.0f} min"
                              "<br>1-in-10 worst: %{customdata[1]:.0f}+ min<br>%{customdata[2]:,} passengers<extra></extra>")
    fig.add_scatter(y=g["stop_direction"], x=g["p90"], mode="markers", name="Worst 10% wait at least…",
                    marker=dict(symbol="line-ns-open", size=18, color=RED, line_width=3))
    fig.update_xaxes(title="minutes")
    return style(fig, 470, "Average wait at each stop and direction",
                 "Bars = average; red tick = the wait the unluckiest 1 in 10 passengers faced")


def fig_boarding(c: pd.DataFrame, thr: int = 10) -> go.Figure:
    rows = []
    for route, grp in c.groupby("stop_direction"):
        n = len(grp)
        k = int(grp["wait_minutes"].le(thr).sum())
        lo, hi = D.wilson_interval(k, n)
        rows.append({"route": route, "share": k / n, "lo": lo, "hi": hi, "n": n})
    bp = pd.DataFrame(rows).sort_values("share")
    fig = go.Figure(go.Bar(
        y=bp["route"], x=bp["share"], orientation="h",
        marker=dict(color=bp["share"], colorscale=[[0, RED], [0.5, AMBER], [1, TEAL]], cmin=0.4, cmax=1),
        error_x=dict(type="data", symmetric=False, array=bp["hi"] - bp["share"],
                     arrayminus=bp["share"] - bp["lo"], color="#555", thickness=1.4),
        text=bp["share"].map("{:.0%}".format), textposition="inside", insidetextanchor="start",
        textfont=dict(color="white", size=12),
        customdata=bp[["n", "lo", "hi"]],
        hovertemplate="%{y}<br>%{x:.1%} boarded within " + str(thr) +
                      " min<br>95% range %{customdata[1]:.0%}–%{customdata[2]:.0%} · n=%{customdata[0]:,}<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%", range=[0, 1.03], title=f"share of passengers boarded within {thr} minutes")
    return style(fig, 470, f"Chance of boarding within {thr} minutes",
                 "Black whiskers show the 95% confidence range")


def fig_block_box(c: pd.DataFrame) -> go.Figure:
    fig = px.box(c, x="Block", y="wait_minutes", color="Block", points=False,
                 category_orders={"Block": ["AM", "MD", "PM"]}, color_discrete_map=BLOCK_COLORS,
                 labels={"wait_minutes": "minutes waited", "Block": ""})
    fig.update_xaxes(ticktext=["Morning (before 11:00)", "Midday (11:00–15:00)", "Evening (after 15:00)"],
                     tickvals=["AM", "MD", "PM"])
    fig.update_layout(showlegend=False)
    return style(fig, 400, "Waits by time of day")


def fig_wait_by_hour(c: pd.DataFrame) -> go.Figure:
    c = c[c["hour_slot"].le(LAST_HOUR)]
    hr = c.groupby(["hour_slot", "Type used"])["wait_minutes"].median().reset_index()
    fig = px.line(hr, x="hour_slot", y="wait_minutes", color="Type used", markers=True,
                  color_discrete_map=TYPE_COLORS,
                  labels={"hour_slot": "hour the passenger joined the queue",
                          "wait_minutes": "typical (median) wait, min", "Type used": "Type"})
    fig.update_xaxes(dtick=1)
    return style(fig, 400, "Typical wait by hour of arrival")


def fig_queue_rhythm(q: pd.DataFrame) -> go.Figure:
    q = q[q["hour_slot"].le(LAST_HOUR)]
    g = q.groupby("hour_slot").agg(mean_queue=("Queue size", "mean"),
                                   arrivals=("New arrivals (10 min)", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=g["hour_slot"], y=g["arrivals"] * 6, name="New passengers arriving per hour (avg per stop)",
                marker_color="rgba(217,154,0,.55)")
    fig.add_scatter(x=g["hour_slot"], y=g["mean_queue"], name="People standing in the queue (avg, right axis)",
                    mode="lines+markers", line=dict(color=TEAL, width=3), yaxis="y2")
    fig.update_layout(yaxis2=dict(title="people in queue", overlaying="y", side="right", rangemode="tozero",
                                  showgrid=False, color=TEAL))
    fig.update_xaxes(title="hour of day", dtick=1)
    fig.update_layout(yaxis=dict(title="new passengers per hour", rangemode="tozero"))
    fig.update_layout(yaxis2=dict(tickformat=".0f"))
    top = g.loc[g["arrivals"].idxmax()]
    fig.add_annotation(x=top["hour_slot"], y=top["arrivals"] * 6, text=f"busiest: {top['arrivals'] * 6:.0f}/hour at {int(top['hour_slot']):02d}:00",
                       showarrow=True, arrowhead=2, ay=-32, font=dict(size=12, color=INK), bgcolor="rgba(255,255,255,.85)")
    return style(fig, 430, "The daily rhythm", "Demand (bars, left axis) and queue length (line, right axis) through the day")


def queue_by_route_hour(q: pd.DataFrame) -> pd.DataFrame:
    q = q[q["hour_slot"].le(LAST_HOUR)]
    return (q.groupby(["stop_direction", "hour_slot"])
            .agg(mean_queue=("Queue size", "mean"), max_queue=("Queue size", "max"),
                 n=("Queue size", "count")).reset_index())


def fig_queue_heat(q: pd.DataFrame) -> go.Figure:
    per = queue_by_route_hour(q)
    order = per.groupby("stop_direction")["mean_queue"].max().sort_values(ascending=False).index
    heat = per.pivot(index="stop_direction", columns="hour_slot", values="mean_queue").loc[order]
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd", text_auto=".0f",
                    labels=dict(x="hour of day", y="", color="avg queue"))
    fig.update_xaxes(dtick=1)
    return style(fig, 480, "Where and when queues build", "Average number of people waiting, by stop-direction and hour")


def fig_rush(q: pd.DataFrame, routes: list[str]) -> go.Figure:
    per = queue_by_route_hour(q)
    fig = px.line(per[per["stop_direction"].isin(routes)], x="hour_slot", y="mean_queue",
                  color="stop_direction", markers=True,
                  labels={"hour_slot": "hour of day", "mean_queue": "average queue (people)", "stop_direction": ""})
    fig.update_xaxes(dtick=1)
    return style(fig, 420, "Rush-hour profile of the busiest routes")


def fig_weekday_weekend(q: pd.DataFrame) -> go.Figure:
    q = q[q["hour_slot"].le(LAST_HOUR)]
    hourly = (q.groupby(["Date", "is_weekend", "stop_direction", "hour_slot"])["New arrivals (10 min)"]
              .mean().mul(6).reset_index(name="arr"))
    hourly["Day type"] = np.where(hourly["is_weekend"], "Weekend", "Weekday")
    agg = hourly.groupby(["Day type", "hour_slot"])["arr"].mean().reset_index()
    fig = px.line(agg, x="hour_slot", y="arr", color="Day type", markers=True,
                  color_discrete_map={"Weekday": TEAL, "Weekend": AMBER},
                  labels={"hour_slot": "hour of day", "arr": "passengers arriving per hour (per route)"})
    fig.update_xaxes(dtick=1)
    return style(fig, 400, "Weekday vs weekend demand")


def fig_friday(q: pd.DataFrame) -> go.Figure:
    sub = q[q["Stop"].isin(["NEW MARKET", "HALL"]) & q["hour"].between(13, 14.5)
            & ~q["is_weekend"]].copy()
    sub["Day group"] = np.where(sub["Day"].eq("Friday"), "Friday", "Mon–Thu")
    g = sub.groupby(["Stop", "Day group"])["Queue size"].mean().reset_index()
    fig = px.bar(g, x="Stop", y="Queue size", color="Day group", barmode="group", text_auto=".1f",
                 color_discrete_map={"Friday": RED, "Mon–Thu": "#9fb4b8"},
                 labels={"Queue size": "average queue, 13:00–14:30", "Stop": ""})
    return style(fig, 380, "The Friday early-afternoon surge")


def left_behind_flags(vv: pd.DataFrame, qq: pd.DataFrame, slack: int = 5) -> pd.Series:
    """True where a vehicle's 'left behind' count is larger than the nearest queue counts
    allow (more people left behind than were in the queue): a likely data-entry error."""
    flag = pd.Series(False, index=vv.index)
    if qq.empty or vv.empty:
        return flag
    qt, qs = qq["Time minutes"].to_numpy(float), qq["Queue size"].to_numpy(float)
    i = np.clip(np.searchsorted(qt, vv["Arrived minutes"].to_numpy(float)), 0, len(qt) - 1)
    qmax = np.maximum(qs[i], qs[np.clip(i - 1, 0, len(qt) - 1)])
    return pd.Series(vv["left_n"].to_numpy(float) > qmax + slack, index=vv.index)


@st.cache_data(show_spinner=False)
def left_behind_check() -> tuple[int, int]:
    n_bad = n_all = 0
    for (d, r), vg in V_ALL.groupby(["Date", "stop_direction"]):
        vv = vehicle_metrics(vg.dropna(subset=["Arrived minutes"]).sort_values("Arrived minutes"))
        qq = Q_ALL[(Q_ALL["Date"] == d) & (Q_ALL["stop_direction"] == r)].sort_values("Time minutes")
        n_bad += int(left_behind_flags(vv, qq).sum())
        n_all += len(vv)
    return n_bad, n_all


def fig_vehicle_day(v: pd.DataFrame, q: pd.DataFrame, date, route: str) -> go.Figure:
    vv = vehicle_metrics(v[(v["Date"].dt.date == date) & (v["stop_direction"] == route)])
    vv = vv.dropna(subset=["Arrived minutes"]).sort_values("Arrived minutes")
    t = pd.to_datetime(vv["Arrived minutes"], unit="m", origin=pd.Timestamp(date))
    fig = go.Figure()
    for typ, color in TYPE_COLORS.items():
        m = vv["Type"].eq(typ)
        fig.add_bar(x=t[m], y=vv.loc[m, "boarded_n"], name=f"{typ}: boarded", marker_color=color,
                    customdata=vv.loc[m, ["Vehicle ID", "Arrived", "Departed", "cap"]],
                    hovertemplate="%{customdata[0]} arrived %{customdata[1]}, left %{customdata[2]}"
                                  "<br>boarded %{y} of %{customdata[3]:.0f} seats<extra></extra>")
    qq = q[(q["Date"].dt.date == date) & (q["stop_direction"] == route)].sort_values("Time minutes")
    flag = left_behind_flags(vv, qq)
    fig.add_bar(x=t[~flag], y=-vv.loc[~flag, "left_n"], name="left behind at the kerb", marker_color="rgba(200,85,61,.55)",
                hovertemplate="%{customdata} left behind<extra></extra>", customdata=vv.loc[~flag, "left_n"])
    if flag.any():
        fig.add_bar(x=t[flag], y=-vv.loc[flag, "left_n"], name="left behind > people in queue (likely entry error)",
                    marker_color="rgba(90,90,90,.55)", marker_pattern_shape="/",
                    hovertemplate="%{customdata} left behind, more than were queuing<extra></extra>", customdata=vv.loc[flag, "left_n"])
    if len(qq):
        fig.add_scatter(x=pd.to_datetime(qq["Time minutes"], unit="m", origin=pd.Timestamp(date)),
                        y=qq["Queue size"], name="people in the queue (counted)", mode="lines",
                        line=dict(color=INK, width=2, dash="dot"))
    fig.update_layout(barmode="relative", bargap=0.05)
    fig.update_xaxes(title="time of day", tickformat="%H:%M")
    fig.update_yaxes(title="people  (↑ boarded · ↓ left behind)")
    return style(fig, 460, f"A day of vehicle arrivals: {route.replace(' -> ', ' → ')}",
                 f"{pd.Timestamp(date):%A %d %B %Y} · each bar is one vehicle pulling in")


def fig_load(load: pd.DataFrame) -> go.Figure:
    fig = px.bar(load.sort_values("load_factor_pct"), y="stop_direction", x="load_factor_pct", color="Type",
                 barmode="group", orientation="h", color_discrete_map=TYPE_COLORS,
                 labels={"load_factor_pct": "average % of seats filled when leaving", "stop_direction": ""})
    fig.add_vline(x=100, line_dash="dot", line_color=RED, annotation_text="full", annotation_position="top")
    return style(fig, 480, "How full are vehicles when they leave?")


def fig_left_behind(load: pd.DataFrame) -> go.Figure:
    fig = px.bar(load.sort_values("left_behind_rate"), y="stop_direction", x="left_behind_rate", color="Type",
                 barmode="group", orientation="h", color_discrete_map=TYPE_COLORS,
                 labels={"left_behind_rate": "left behind ÷ (boarded + left behind)", "stop_direction": ""})
    fig.update_xaxes(tickformat=".0%")
    return style(fig, 480, "How often are people left behind?")


def fig_headway(headway: pd.DataFrame) -> go.Figure:
    hw = headway.dropna(subset=["median_headway_min"])
    hw = hw[hw["hour_slot"].le(LAST_HOUR)]
    heat = hw.pivot(index="stop_direction", columns="hour_slot", values="median_headway_min")
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues", text_auto=".0f",
                    labels=dict(x="hour of day", y="", color="minutes"))
    fig.update_xaxes(dtick=1)
    return style(fig, 440, "Minutes between vehicles (headway)", "Typical gap between one vehicle and the next")


def fig_dc(dc: pd.DataFrame) -> go.Figure:
    dc = dc.dropna(subset=["demand_to_capacity_ratio"])
    dc = dc[dc["hour_slot"].le(LAST_HOUR)]
    heat = dc.pivot(index="stop_direction", columns="hour_slot", values="demand_to_capacity_ratio")
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn_r", zmin=0, zmax=2, text_auto=".1f",
                    labels=dict(x="hour of day", y="", color="ratio"))
    fig.update_xaxes(dtick=1)
    return style(fig, 460, "Passengers arriving ÷ seats leaving", "Above 1.0 (red) = more people turn up than there are seats")


def fig_effects(t: pd.DataFrame) -> go.Figure:
    t = t.dropna(subset=["effect_size"]).copy()
    t["label"] = t["question"] + " · " + t["test_name"]
    t = t.iloc[::-1]
    sig = t["p_value"].lt(0.05)
    fig = go.Figure(go.Scatter(
        x=t["effect_size"], y=t["label"], mode="markers",
        marker=dict(size=13, color=np.where(sig, TEAL, "#aaaaaa")),
        error_x=dict(type="data", symmetric=False,
                     array=(t["effect_size_ci95_high"] - t["effect_size"]).fillna(0),
                     arrayminus=(t["effect_size"] - t["effect_size_ci95_low"]).fillna(0)),
        customdata=t[["effect_size_name", "p_value"]],
        hovertemplate="%{y}<br>%{customdata[0]} = %{x:.2f} · p = %{customdata[1]:.2g}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#999")
    for x, lab in ((0.1, "small"), (0.3, "medium"), (0.5, "large")):
        fig.add_vline(x=x, line_dash="dot", line_color="#ddd", annotation_text=lab, annotation_position="top")
    fig.update_xaxes(title="effect size (0 = no difference, ±1 = complete separation)", range=[-0.45, 1])
    return style(fig, 470, "How big is each difference?", "Dots = effect size; bars = 95% confidence range")


def fig_spearman(sp: pd.DataFrame) -> go.Figure:
    names = {"queue": "queue size", "arrivals": "arrivals", "wait": "waiting time", "hour_slot": "hour of day",
             "load_factor": "how full vehicles leave", "headway": "gap between vehicles"}
    sp = sp.set_index("variable").rename(index=names, columns=names)
    sp.index.name = None
    fig = px.imshow(sp, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    return style(fig, 430, "Which measures move together?", "Spearman correlation: +1 rise together, −1 opposite, 0 unrelated")


def fig_perceived(comp: pd.DataFrame) -> go.Figure:
    comp = comp.dropna(subset=["observed_wait_mean_min"])
    fig = go.Figure()
    fig.add_bar(x=comp["boarding_stop"], y=comp["reported_wait_mean_min"], name="What students said (survey)",
                marker_color=AMBER, customdata=comp["reported_wait_n"],
                hovertemplate="%{x}: %{y:.1f} min (%{customdata} students)<extra></extra>")
    fig.add_bar(x=comp["boarding_stop"], y=comp["observed_wait_mean_min"], name="What we timed (field)",
                marker_color=TEAL, customdata=comp["observed_wait_n"],
                hovertemplate="%{x}: %{y:.1f} min (%{customdata:,} timed waits)<extra></extra>")
    fig.update_yaxes(title="average wait (min)")
    return style(fig, 400, "Perceived vs measured waiting time")


def fig_survey_bar(s: pd.DataFrame, col: str, title: str, order: list[str] | None = None) -> go.Figure:
    vc = s[col].value_counts()
    if order:
        vc = vc.reindex(order).dropna()
    df = vc.reset_index()
    df.columns = ["answer", "count"]
    df["share"] = df["count"] / df["count"].sum()
    fig = px.bar(df, x="count", y="answer", orientation="h", text=df["share"].map("{:.0%}".format),
                 color_discrete_sequence=[TEAL], labels={"answer": "", "count": "students"})
    fig.update_yaxes(categoryorder="array", categoryarray=list(df["answer"])[::-1])
    return style(fig, 330, title)


def fig_multi(s: pd.DataFrame, col: str, title: str) -> go.Figure:
    ex = s[col].dropna().str.split(r",\s*").explode().str.strip()
    vc = ex[ex.ne("")].value_counts().head(8).reset_index()
    vc.columns = ["option", "count"]
    vc["share"] = vc["count"] / len(s)
    fig = px.bar(vc.iloc[::-1], x="count", y="option", orientation="h", text=vc["share"].iloc[::-1].map("{:.0%}".format),
                 color_discrete_sequence=[AMBER], labels={"option": "", "count": "students choosing it"})
    fig.update_yaxes(dtick=1, automargin=True)
    return style(fig, 360, title, "Students could pick more than one")


def fleet_daily(fl: pd.DataFrame, q: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    op = fl[fl["Status"].eq("Operating")].groupby(["Date", "Type"]).size().unstack(fill_value=0)
    d = pd.DataFrame({"CNG bus": op.get("CNG bus", 0), "Keke": op.get("Keke", 0)})
    d["seats"] = d["CNG bus"] * D.CAPACITIES["CNG bus"] + d["Keke"] * D.CAPACITIES["Keke"]
    d["demand"] = q.groupby("Date")["New arrivals (10 min)"].sum()
    d["per_seat"] = d["demand"] / d["seats"]
    cw = w[w["wait_event"] & w["Date"].ge(D.MAIN_WEEK_START)]
    d["ok10"] = cw.groupby("Date")["wait_minutes"].apply(lambda s: s.le(10).mean())
    d["label"] = pd.to_datetime(d.index).strftime("%a %d")
    d["fleet_cng"] = fl.loc[fl["Type"].eq("CNG bus"), "Vehicle ID"].nunique()
    d["fleet_keke"] = fl.loc[fl["Type"].eq("Keke"), "Vehicle ID"].nunique()
    return d


def fig_fleet_daily(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=d["label"], y=d["CNG bus"], name="CNG buses running", marker_color=TEAL,
                text=d["CNG bus"], textposition="inside")
    fig.add_bar(x=d["label"], y=d["Keke"], name="Keke running", marker_color=AMBER,
                text=d["Keke"], textposition="inside")
    fig.add_scatter(x=d["label"], y=d["ok10"], name="share boarding within 10 min (right axis)", yaxis="y2",
                    mode="lines+markers", line=dict(color=RED, width=3))
    fig.update_layout(barmode="stack",
                      yaxis=dict(title="vehicles seen operating"),
                      yaxis2=dict(title="boarded within 10 min", overlaying="y", side="right", range=[0.5, 1], dtick=0.1,
                                  tickformat=".0%", showgrid=False, color=RED))
    return style(fig, 420, "How many vehicles were on the road each day?",
                 "Vehicles seen at our stops (bars) and how quickly passengers boarded (line)")


def fig_fleet_grid(fl: pd.DataFrame) -> go.Figure:
    g = fl.assign(on=fl["Status"].eq("Operating").astype(int),
                  day=pd.to_datetime(fl["Date"]).dt.strftime("%a %d"))
    order = (g.groupby(["Type", "Vehicle ID"])["on"].sum().reset_index()
             .sort_values(["Type", "on"], ascending=[True, False]))
    piv = g.pivot_table(index="Vehicle ID", columns="day", values="Arrivals logged", aggfunc="sum")
    piv = piv.loc[order["Vehicle ID"], sorted(g["day"].unique(), key=lambda s: pd.to_datetime(
        g.loc[g["day"].eq(s), "Date"].iat[0]))]
    fig = px.imshow(piv.replace(0, np.nan), aspect="auto", color_continuous_scale="Teal", text_auto=".0f",
                    labels=dict(x="", y="", color="arrivals logged"))
    fig.update_traces(hovertemplate="%{y} · %{x}<br>%{z} arrivals logged<extra></extra>")
    return style(fig, 620, "Which vehicles worked which days (sheet D, rebuilt)",
                 "Number = arrivals we logged for that vehicle; blank = not seen at our stops that day")


def fig_validation_days(t: pd.DataFrame) -> go.Figure:
    g = (t.groupby("Date").agg(model=("abs_error", "mean"), base=("baseline_abs_error", "mean"))
         .reset_index())
    g["day"] = pd.to_datetime(g["Date"]).dt.strftime("%a %d")
    fig = go.Figure()
    fig.add_bar(x=g["day"], y=g["base"], name="Naive guess (same wait for everyone)", marker_color="#b9c6c9",
                text=g["base"].map("{:.1f}".format), textposition="outside")
    fig.add_bar(x=g["day"], y=g["model"], name="Our model (stop, direction, time block, vehicle type)",
                marker_color=TEAL, text=g["model"].map("{:.1f}".format), textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="average error (minutes)", range=[0, g[["model", "base"]].max().max() * 1.2])
    return style(fig, 400, "How far off were the predictions?",
                 "Each day was predicted using only the other six days; lower is better")


def fig_calibration(t: pd.DataFrame) -> go.Figure:
    t = t.assign(bin=pd.cut(t["p10"], [0, .5, .7, .85, .95, 1.0], include_lowest=True))
    g = t.groupby("bin", observed=True).agg(pred=("p10", "mean"), act=("boarded_10", "mean"), n=("p10", "size")).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=[0.3, 1], y=[0.3, 1], mode="lines", name="perfect prediction",
                    line=dict(color="#aaa", dash="dot"))
    fig.add_scatter(x=g["pred"], y=g["act"], mode="markers+lines", name="our model",
                    marker=dict(size=np.sqrt(g["n"]) / 2.2, color=TEAL), line=dict(color=TEAL),
                    customdata=g["n"], hovertemplate="predicted %{x:.0%} → actually %{y:.0%}<br>%{customdata:,} passengers<extra></extra>")
    fig.update_xaxes(title="predicted chance of boarding within 10 min", tickformat=".0%", range=[0.3, 1.02])
    fig.update_yaxes(title="share who actually did", tickformat=".0%", range=[0.3, 1.02])
    return style(fig, 400, "When the model says 80%, does it happen 80% of the time?")


# ==========================================================================
# Percentiles, survival curve, tail and simulation (cng_sim)
# ==========================================================================
@st.cache_data(show_spinner=False)
def pct_table(by: tuple[str, ...] | None = None, main_week: bool = True) -> pd.DataFrame:
    w = W_ALL[W_ALL["Date"].ge(D.MAIN_WEEK_START)] if main_week else W_ALL
    return S.percentile_table(w, list(by) if by else None)


@st.cache_data(show_spinner="Running the queue simulation…")
def scenarios(extra: int = 2) -> tuple[pd.DataFrame, dict]:
    q = Q_ALL[Q_ALL["Date"].ge(D.MAIN_WEEK_START)]
    v = V_ALL[V_ALL["Date"].ge(D.MAIN_WEEK_START)]
    return S.scenario_table(q, v, extra)


@st.cache_data(show_spinner="Running the queue simulation…")
def custom_scenario(extra: int, mode: str, spacing: bool) -> pd.DataFrame:
    q = Q_ALL[Q_ALL["Date"].ge(D.MAIN_WEEK_START)]
    v = V_ALL[V_ALL["Date"].ge(D.MAIN_WEEK_START)]
    mods = []
    if extra:
        mods.append(S.scenario_reallocate(S.HOT_CELLS, S.DONORS, extra) if mode == "move"
                    else S.scenario_add_peak(S.HOT_CELLS, extra))
    if spacing:
        mods.append(S.scenario_even_spacing)
    return S.simulate(q, v, S.compose(*mods) if mods else None)


def pct_cut(w: pd.DataFrame, p: float) -> float:
    s = S.survival_input(w)
    return S.km_percentile(S.kaplan_meier(s["duration"], s["event"]), p)


def fig_survival(curves: dict[str, pd.DataFrame], title: str, subtitle: str | None = None,
                 shade_first: bool = True, colors: list[str] | None = None, xmax: int = 40, field: bool = False) -> go.Figure:
    colors = colors or [TEAL, RED, AMBER, "#8a63d2", INK]
    fig = go.Figure()
    for i, (label, w) in enumerate(curves.items()):
        s = field_survival_input(w) if field else S.survival_input(w)
        full = S.kaplan_meier(s["duration"], s["event"])
        surv = full[full["t"] <= xmax]
        area = S.area_under_curve(full)
        p90 = S.km_percentile(full, 90)
        if field:
            last_t = int(s["duration"].max())
            full = full[full["t"] <= last_t]
            surv = full[full["t"] <= xmax]
        fill = "tozeroy" if (shade_first and i == 0) or len(curves) > 1 else None
        fig.add_scatter(x=surv["t"], y=surv["still_waiting"], mode="lines", line_shape="hv",
                        name=(f"{label}  ·  P90 {p90:.0f} min · area {area:.1f}" if not np.isnan(p90)
                              else f"{label}  ·  P90 longer than observed"),
                        line=dict(color=colors[i % len(colors)], width=3),
                        fill=fill, fillcolor=(colors[i % len(colors)] + "22") if fill else None,
                        hovertemplate="after %{x} min: %{y:.0%} still waiting<extra>" + label + "</extra>")
        if (i == 0 or len(curves) > 1) and not np.isnan(p90) and p90 <= xmax:
            fig.add_vline(x=p90, line_dash="dot", line_color=colors[i % len(colors)],
                          annotation_text=f"P90 {p90:.0f}", annotation_position="top", annotation_font_color=colors[i % len(colors)])
    fig.add_hline(y=0.10, line_dash="dot", line_color="#999",
                  annotation_text="10% still waiting = P90", annotation_position="top right")
    fig.add_hline(y=0.50, line_dash="dot", line_color="#ccc",
                  annotation_text="50% = typical wait (P50)", annotation_position="top right")
    fig.update_xaxes(title="minutes since joining the queue", range=[0, xmax], dtick=5)
    fig.update_yaxes(title="share of passengers still waiting", tickformat=".0%", range=[0, 1.02])
    fig.update_layout(legend=dict(orientation="v", y=0.98, x=0.42, yanchor="top", bgcolor="rgba(255,255,255,.85)"))
    return style(fig, 440, title, subtitle)


def fig_ladder(t: pd.DataFrame, label: str, title: str, subtitle: str | None = None, target: float | None = None) -> go.Figure:
    t = t.sort_values("P90").copy()
    t[label] = t[label].astype(str).str.replace(" -> ", " → ")
    fig = go.Figure()
    for _, r in t.iterrows():
        fig.add_scatter(x=[r["P50"], r["P95"]], y=[r[label]] * 2, mode="lines",
                        line=dict(color="#c9d4d6", width=8), showlegend=False, hoverinfo="skip")
    for p, color, sym in (("P50", TEAL, "circle"), ("P75", AMBER, "circle"), ("P90", RED, "diamond"), ("P95", INK, "line-ns-open")):
        fig.add_scatter(x=t[p], y=t[label], mode="markers", name=p.replace("P", "") + "th percentile" if p != "P50" else "50th (typical)",
                        marker=dict(color=color, size=13 if p != "P95" else 16, symbol=sym, line=dict(width=2 if p == "P95" else 0, color=color)),
                        customdata=t[["n"]], hovertemplate="%{y}<br>" + p + " = %{x:.0f} min · n=%{customdata[0]:,}<extra></extra>")
    if target:
        fig.add_vline(x=target, line_dash="dash", line_color=RED, annotation_text=f"{target:.0f}-min P90 target", annotation_position="bottom right")
    fig.update_xaxes(title="minutes waited")
    return style(fig, max(320, 46 * len(t) + 140), title, subtitle)


def fig_bands(c: pd.DataFrame) -> go.Figure:
    wv = c["wait_minutes"].dropna()
    cuts = np.percentile(wv, [50, 75, 90])
    bands = pd.cut(wv.rank(method="first", pct=True), [0, .5, .75, .9, 1.0],
                   labels=["Fastest 50%", "Next 25%", "Next 15%", "Slowest 10%"])
    g = pd.DataFrame({"band": bands, "w": wv}).groupby("band", observed=True)["w"].agg(["size", "sum"])
    g["share_pax"] = g["size"] / g["size"].sum()
    g["share_min"] = g["sum"] / g["sum"].sum()
    g = g.reset_index()
    fig = go.Figure()
    fig.add_bar(x=g["band"], y=g["share_pax"], name="share of passengers", marker_color="#b9c6c9",
                text=g["share_pax"].map("{:.0%}".format), textposition="outside")
    fig.add_bar(x=g["band"], y=g["share_min"], name="share of all minutes spent waiting", marker_color=RED,
                text=g["share_min"].map("{:.0%}".format), textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", range=[0, max(g["share_pax"].max(), g["share_min"].max()) * 1.2], title="")
    fig.update_xaxes(ticktext=[f"Fastest 50%<br>(≤{cuts[0]:.0f} min)", f"Next 25%<br>({cuts[0]:.0f}–{cuts[1]:.0f} min)",
                               f"Next 15%<br>({cuts[1]:.0f}–{cuts[2]:.0f} min)", f"Slowest 10%<br>(>{cuts[2]:.0f} min)"],
                     tickvals=list(g["band"]))
    return style(fig, 400, "Who carries the waiting?", "Passengers vs passenger-minutes, by percentile band")


def tail_sources(c: pd.DataFrame) -> pd.DataFrame:
    cut = np.percentile(c["wait_minutes"].dropna(), 90)
    t = c[c["wait_minutes"] > cut]
    g = (t.groupby(["stop_direction", "Block"])["wait_minutes"].agg(["size", "sum"]).reset_index())
    g["share"] = g["sum"] / g["sum"].sum()
    allp = c.groupby(["stop_direction", "Block"]).size().rename("all_pax").reset_index()
    g = g.merge(allp, on=["stop_direction", "Block"])
    g["pax_share"] = g["all_pax"] / len(c)
    g["label"] = g["stop_direction"].str.replace(" -> ", " → ") + " · " + g["Block"].map({"AM": "morning", "MD": "midday", "PM": "evening"})
    return g.sort_values("share", ascending=False)


def fig_tail_sources(c: pd.DataFrame) -> go.Figure:
    g = tail_sources(c).head(8).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=g["label"], x=g["share"], orientation="h", name="share of slowest-10% waiting minutes",
                marker_color=RED, text=g["share"].map("{:.0%}".format), textposition="outside")
    fig.add_bar(y=g["label"], x=g["pax_share"], orientation="h", name="share of all passengers",
                marker_color="#b9c6c9", text=g["pax_share"].map("{:.0%}".format), textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%", range=[0, g["share"].max() * 1.25])
    return style(fig, 470, "Where the slowest 10% of waits happen", "Red = share of the tail’s waiting minutes; grey = share of all passengers")


def fig_sim_check(obs: pd.DataFrame, sim: pd.DataFrame) -> go.Figure:
    m = obs.merge(sim, on=["stop_direction", "Block"], suffixes=("_o", "_s"))
    m["label"] = m["stop_direction"].str.replace(" -> ", " → ") + " · " + m["Block"]
    hi = float(max(m["P90_o"].max(), m["P90_s"].max())) + 3
    fig = go.Figure()
    fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", name="perfect match", line=dict(color="#aaa", dash="dot"))
    fig.add_scatter(x=m["P90_o"], y=m["P90_s"], mode="markers", name="stop-direction × time block",
                    marker=dict(size=11, color=m["Block"].map(BLOCK_COLORS)), text=m["label"],
                    hovertemplate="%{text}<br>observed P90 %{x:.0f} · simulated %{y:.0f}<extra></extra>")
    fig.update_xaxes(title="observed P90 (timed passengers, sheet B2)", range=[0, hi])
    fig.update_yaxes(title="simulated P90", range=[0, hi])
    return style(fig, 420, "Does the simulation reproduce reality?", "Each dot = one stop-direction in one time block")


def fig_scenarios(t: pd.DataFrame) -> go.Figure:
    t = t.copy()
    short = {t["scenario"].iat[0]: "As observed", t["scenario"].iat[1]: "Even spacing<br>(same vehicles)",
             t["scenario"].iat[2]: "Move 2 buses/hr<br>to peaks", t["scenario"].iat[3]: "Add 2 buses/hr<br>at peaks",
             t["scenario"].iat[4]: "Move 2 buses/hr<br>+ even spacing"}
    t["s"] = t["scenario"].map(short)
    fig = go.Figure()
    fig.add_bar(x=t["s"], y=t["P90"], name="P90 wait (min)", marker_color=RED, text=t["P90"].map("{:.0f}".format), textposition="outside")
    fig.add_bar(x=t["s"], y=t["area"], name="area under curve = average wait (min)", marker_color=TEAL,
                text=t["area"].map("{:.1f}".format), textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="minutes", range=[0, t["P90"].max() * 1.2])
    return style(fig, 420, "What each option does to the 90th percentile and the area", "Simulated over the full observation week")


# ==========================================================================
# Field records, model checks, day-level analysis, departure waits, route map
# ==========================================================================
def field_survival_input(w: pd.DataFrame) -> pd.DataFrame:
    """Field passengers still waiting are censored at the end of THAT field session
    (the last clock time recorded for that date and route), not at 18:00."""
    w = w.copy()
    key = [w["Date"].fillna(pd.Timestamp("1900-01-01")), w["stop_direction"]]
    end = pd.concat([w["joined_minutes"], w["boarded_minutes"]], axis=1).max(axis=1).groupby(key).transform("max")
    dur = w["wait_minutes"].copy()
    cens = w["censored"] & w["joined_minutes"].notna()
    dur[cens] = (end[cens] - w.loc[cens, "joined_minutes"]).clip(lower=0)
    w["duration"], w["event"] = dur, ~w["censored"]
    return w.dropna(subset=["duration"])


def km_from(si: pd.DataFrame) -> pd.DataFrame:
    return S.kaplan_meier(si["duration"], si["event"])


def fig_field_waits(w: pd.DataFrame) -> go.Figure:
    si = field_survival_input(w)
    t = pd.to_datetime(si["joined_minutes"], unit="m", origin=pd.Timestamp("2026-09-27"))
    done, still = si[si["event"]], si[~si["event"]]
    fig = go.Figure()
    fig.add_scatter(x=t[done.index], y=done["duration"], mode="markers", name="boarded (minutes waited)",
                    marker=dict(size=11, color=TEAL), hovertemplate="joined %{x|%H:%M} · waited %{y:.0f} min<extra></extra>")
    fig.add_scatter(x=t[still.index], y=still["duration"], mode="markers", name="still waiting when we stopped (at least this long)",
                    marker=dict(size=12, color=RED, symbol="triangle-up-open", line=dict(width=2)),
                    hovertemplate="joined %{x|%H:%M} · still waiting after %{y:.0f}+ min<extra></extra>")
    fig.update_xaxes(title="time the passenger joined the queue", tickformat="%H:%M")
    fig.update_yaxes(title="minutes waited", rangemode="tozero")
    return style(fig, 400, "Every passenger we timed by hand",
                 "Sun 27 Sept · Bus Stop → Campus Gate · CNG bus · each dot is one person")


def fig_field_vs_model_load(vf: pd.DataFrame, vm: pd.DataFrame) -> go.Figure:
    f = vehicle_metrics(vf).groupby("stop_direction")["load"].agg(["mean", "size"]).rename(columns={"mean": "field", "size": "n_field"})
    m = vehicle_metrics(vm).groupby("stop_direction")["load"].mean().rename("model")
    g = f.join(m, how="inner").dropna(subset=["field"]).reset_index().sort_values("field")
    g["label"] = g["stop_direction"].str.replace(" -> ", " → ")
    fig = go.Figure()
    fig.add_bar(y=g["label"], x=g["field"], orientation="h", name="field records", marker_color=TEAL,
                text=g.apply(lambda r: f"{r['field']:.0%} (n={r['n_field']})", axis=1), textposition="outside")
    fig.add_bar(y=g["label"], x=g["model"], orientation="h", name="modelled week", marker_color="#e3c46d",
                text=g["model"].map("{:.0%}".format), textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.25], title="average share of seats filled when leaving")
    return style(fig, 380, "How full vehicles left: field vs modelled week")


def fig_field_queues(q: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (label, sub), color in zip(
            [("Bus Stop → Campus Gate · Tue 27 Oct", q[q["stop_direction"].eq("BUS STOP -> CAMPUS GATE")]),
             ("SUB → Campus Gate · date not written on sheet", q[q["stop_direction"].eq("SUB -> CAMPUS GATE")])], [TEAL, RED]):
        sub = sub.sort_values("Time minutes")
        if sub.empty:
            continue
        fig.add_scatter(x=pd.to_datetime(sub["Time minutes"], unit="m", origin=pd.Timestamp("2026-10-27")),
                        y=sub["Queue size"], mode="lines+markers", name=label, line=dict(color=color, width=2.5), marker=dict(size=6),
                        hovertemplate="%{x|%H:%M}: %{y} people queuing<extra></extra>")
    fig.update_xaxes(title="time of day", tickformat="%H:%M")
    fig.update_yaxes(title="people in the queue", rangemode="tozero")
    return style(fig, 380, "Queue counts taken by hand", "Each point is one count by an observer pair")


@st.cache_data(show_spinner=False)
def field_model_table() -> pd.DataFrame:
    """Side-by-side checks of the modelled week against the field records."""
    rows = []
    fv, mv = vehicle_metrics(V_F), vehicle_metrics(V_ALL)
    for r in ("BUS STOP -> CAMPUS GATE", "CAMPUS GATE -> HALL", "HALL -> CAMPUS GATE", "SUB -> NEW MARKET", "CAMPUS GATE -> BUS STOP"):
        f, m = fv[fv["stop_direction"].eq(r)], mv[mv["stop_direction"].eq(r)]
        if f["load"].notna().sum() >= 5:
            rows.append({"Check": f"Seats filled on leaving · {r.replace(' -> ', ' → ').title()}",
                         "Field": f"{f['load'].mean():.0%} (n={f['load'].notna().sum()})", "Modelled week": f"{m['load'].mean():.0%}",
                         "Verdict": "close" if abs(f["load"].mean() - m["load"].mean()) < .1 else ("model lower" if m["load"].mean() < f["load"].mean() else "model higher")})
    bf = fv[fv["stop_direction"].eq("BUS STOP -> CAMPUS GATE")]
    bm = mv[mv["stop_direction"].eq("BUS STOP -> CAMPUS GATE")]
    rows.append({"Check": "Left behind per person boarded · Bus Stop → Campus Gate",
                 "Field": f"{bf['left_n'].sum() / bf['boarded_n'].sum():.1f}", "Modelled week": f"{bm['left_n'].sum() / bm['boarded_n'].sum():.1f}",
                 "Verdict": "both high; model higher"})
    fw = field_survival_input(W_F)
    mm = W_ALL[W_ALL["stop_direction"].eq("BUS STOP -> CAMPUS GATE") & W_ALL["Type used"].eq("CNG bus") & W_ALL["hour"].between(12.5, 15.75)]
    mw = S.survival_input(mm)
    rows.append({"Check": "Still waiting after 10 min · Bus Stop → Campus Gate, CNG, afternoon",
                 "Field": f"{km_from(fw).set_index('t').loc[10, 'still_waiting']:.0%} (n={len(fw)}, Sunday)",
                 "Modelled week": f"{km_from(mw).set_index('t').loc[10, 'still_waiting']:.0%} (all days) · "
                                  f"{km_from(S.survival_input(mm[mm['Day'].eq('Sunday')])).set_index('t').loc[10, 'still_waiting']:.0%} (Sunday)",
                 "Verdict": "model too optimistic"})
    fq = Q_F[Q_F["stop_direction"].eq("SUB -> CAMPUS GATE")]
    if len(fq):
        rows.append({"Check": "Queue · SUB → Campus Gate, 14:30–16:30",
                     "Field": f"{fq[fq['hour'].between(14.5, 16.5)]['Queue size'].mean():.0f} people on average", "Modelled week": "route not modelled",
                     "Verdict": "gap in the model"})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def departure_waits() -> pd.DataFrame:
    """Join → the moment the boarded vehicle actually left (matches each timed passenger
    to the vehicle of the same type that was at the stop when they boarded)."""
    out = []
    mv = V_ALL.dropna(subset=["Arrived minutes"])
    for (d, r, t), g in completed(W_ALL).groupby(["Date", "stop_direction", "Type used"]):
        vv = mv[(mv["Date"] == d) & (mv["stop_direction"] == r) & (mv["Type"] == t)].sort_values("Arrived minutes")
        if vv.empty:
            continue
        arr = vv["Arrived minutes"].to_numpy(float)
        dep = vv["Departed minutes"].fillna(vv["Arrived minutes"]).to_numpy(float)
        b = g["boarded_minutes"].to_numpy(float)
        i = np.searchsorted(arr, b, side="right") - 1
        d_ = np.where(i >= 0, dep[np.clip(i, 0, None)], np.nan)
        d_ = np.where(d_ < b, np.nan, d_)
        out.append(pd.DataFrame({"Type used": t, "to_board": g["wait_minutes"].to_numpy(), "to_depart": d_ - g["joined_minutes"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner=False)
def daily_table() -> pd.DataFrame:
    w = W_ALL[W_ALL["Date"].ge(D.MAIN_WEEK_START)]
    t = S.percentile_table(w, ["Date"])
    c = completed(w).groupby("Date")["wait_minutes"].apply(lambda s: s.le(10).mean()).rename("ok10")
    t = t.merge(c, left_on="Date", right_index=True)
    t["Day"] = pd.to_datetime(t["Date"]).dt.day_name()
    t["weekend"] = t["Day"].isin(["Saturday", "Sunday"])
    return t.sort_values("Date")


def permutation_p(values: pd.Series, groups: pd.Series) -> float:
    """Exact two-sided permutation test on the difference in group means (days as the unit)."""
    from itertools import combinations
    v = values.to_numpy(float)
    g = groups.to_numpy(bool)
    k, n = int(g.sum()), len(v)
    obs = abs(v[g].mean() - v[~g].mean())
    hits = total = 0
    for idx in combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        total += 1
        hits += abs(v[m].mean() - v[~m].mean()) >= obs - 1e-9
    return hits / total


def fig_daily(t: pd.DataFrame) -> go.Figure:
    t = t.copy()
    t["label"] = pd.to_datetime(t["Date"]).dt.strftime("%a %d")
    colors = np.where(t["weekend"], AMBER, np.where(t["Day"].eq("Friday"), RED, TEAL))
    fig = go.Figure()
    fig.add_bar(x=t["label"], y=t["P90"], marker_color=colors, text=t["P90"].map("{:.0f}".format), textposition="outside",
                name="P90 wait", customdata=t[["P50", "area", "n"]],
                hovertemplate="%{x}<br>P90 %{y:.0f} · P50 %{customdata[0]:.0f} · average %{customdata[1]:.1f} min<br>n=%{customdata[2]:,}<extra></extra>")
    fig.update_yaxes(title="P90 wait (minutes)", range=[0, t["P90"].max() * 1.25])
    fig.update_layout(showlegend=False)
    return style(fig, 360, "Each day is one data point", "P90 wait per day of the modelled week · teal weekdays, red Friday, amber weekend")


def fig_dep_waits(dep: pd.DataFrame) -> go.Figure:
    g = dep.groupby("Type used")[["to_board", "to_depart"]].median().reindex(["CNG bus", "Keke"])
    fig = go.Figure()
    fig.add_bar(x=g.index, y=g["to_board"], name="wait to board", marker_color=["#8fbcc1", "#f0d58f"],
                text=g["to_board"].map("{:.0f} min".format), textposition="outside")
    fig.add_bar(x=g.index, y=g["to_depart"], name="wait until the vehicle leaves", marker_color=[TEAL, AMBER],
                text=g["to_depart"].map("{:.0f} min".format), textposition="outside")
    fig.update_layout(barmode="group", showlegend=True)
    fig.update_yaxes(title="typical (median) minutes", range=[0, g.max().max() * 1.35])
    return style(fig, 400, "Waiting to board vs waiting to leave", "Modelled week · median per passenger")


ROUTE_NODES = {
    "BUS STOP": (0.0, 1.0), "CAMPUS GATE": (2.0, 1.0), "ACADEMICS": (2.7, 3.0), "HALL": (4.2, 2.2),
    "NEW MARKET": (3.7, -0.3), "SUB": (5.7, 1.0),
}


def fig_route_map(tbl: pd.DataFrame, demand: pd.Series) -> go.Figure:
    """Schematic (not to scale) map of the observed stop-directions, coloured by P90 wait."""
    fig = go.Figure()
    def colour(p90):
        if pd.isna(p90):
            return "#b9c6c9"
        return TEAL if p90 <= 10 else (AMBER if p90 <= 15 else RED)
    tbl = tbl.set_index("stop_direction")
    dmax = float(demand.max()) if len(demand) else 1
    routes = [r for r in Q_ALL["stop_direction"].dropna().unique()]
    for r in routes:
        a_, b_ = [s.strip() for s in r.split("->")]
        if a_ not in ROUTE_NODES or b_ not in ROUTE_NODES:
            continue
        (x0, y0), (x1, y1) = ROUTE_NODES[a_], ROUTE_NODES[b_]
        dx, dy = x1 - x0, y1 - y0
        L = (dx ** 2 + dy ** 2) ** .5
        ox, oy = -dy / L * 0.09, dx / L * 0.09  # offset so the two directions sit side by side
        sx, sy, ex, ey = x0 + ox + dx * .13, y0 + oy + dy * .13, x1 + ox - dx * .13, y1 + oy - dy * .13
        p90 = tbl["P90"].get(r, np.nan)
        p50 = tbl["P50"].get(r, np.nan)
        width = 2 + 10 * float(demand.get(r, 0)) / dmax
        fig.add_annotation(x=ex, y=ey, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y", text="",
                           showarrow=True, arrowhead=2, arrowsize=0.6, arrowwidth=width, arrowcolor=colour(p90), opacity=.9)
        label = "no timed waits" if pd.isna(p90) else f"P50 {p50:.0f} · P90 {p90:.0f} min"
        fig.add_scatter(x=[(sx + ex) / 2], y=[(sy + ey) / 2], mode="markers", marker=dict(size=18, opacity=0),
                        showlegend=False, hovertemplate=f"<b>{r.replace(' -> ', ' → ')}</b><br>{label}<br>"
                                                        f"{demand.get(r, 0):,.0f} passengers/day<extra></extra>")
    label_shift = {"BUS STOP": (-46, 0), "CAMPUS GATE": (0, -30), "ACADEMICS": (0, 26), "HALL": (34, 18),
                   "NEW MARKET": (0, -28), "SUB": (30, 20)}
    for name, (x, y) in ROUTE_NODES.items():
        fig.add_scatter(x=[x], y=[y], mode="markers", marker=dict(size=22, color="white", line=dict(color=INK, width=2.5)),
                        showlegend=False, hoverinfo="skip")
        dx_, dy_ = label_shift.get(name, (0, 24))
        fig.add_annotation(x=x, y=y, text=f"<b>{name.title()}</b>", showarrow=False, xshift=dx_, yshift=dy_,
                           font=dict(size=13, color=INK), bgcolor="rgba(251,250,247,.92)", borderpad=2)
    for lab, col in (("P90 ≤ 10 min", TEAL), ("P90 10–15 min", AMBER), ("P90 > 15 min", RED), ("not timed", "#b9c6c9")):
        fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=6), name=lab)
    fig.update_xaxes(visible=False, range=[-0.6, 6.3])
    fig.update_yaxes(visible=False, range=[-0.9, 3.6], scaleanchor="x", scaleratio=1)
    return style(fig, 460, "The network at a glance", "Schematic, not to scale · arrow colour = P90 wait · arrow width = passengers per day")


SURVEY_ORDERS = {
    "typical_wait": ["0 to 5 minutes", "6 to 10 minutes", "11 to 15 minutes", "16 to 30 minutes", "More than 30 minutes"],
    "longest_wait": ["0 to 5 minutes", "6 to 15 minutes", "16 to 30 minutes", "31 to 60 minutes", "More than 60 minutes"],
    "failed_boarding": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "walking_time": ["Less than 1 minute", "1 to 5 minutes", "6 to 10 minutes", "More than 10 minutes"],
    "fare_affordability": ["Very affordable", "Somewhat affordable", "Neither affordable nor expensive",
                           "Somewhat expensive", "Very expensive"],
    "safety": ["Very unsafe", "Unsafe", "Neutral", "Safe", "Very safe"],
    "comfort": ["Very uncomfortable", "Uncomfortable", "Neutral", "Comfortable", "Very comfortable"],
    "usage_frequency": ["Daily", "Several times a week", "Once a week", "Occasionally"],
}
SURVEY_Q = {
    "typical_wait": "Typical wait before boarding", "longest_wait": "Longest wait ever experienced",
    "failed_boarding": "How often the first vehicle is too full to board", "walking_time": "Walk to the nearest stop",
    "fare_affordability": "Fare affordability", "safety": "How safe they feel", "comfort": "Comfort",
    "tracking_app": "Would use a vehicle-tracking app", "transport_used": "Transport type used",
    "transport_preference": "Preferred transport type", "usage_frequency": "How often they ride",
    "boarding_stop": "Usual boarding stop", "destination": "Usual destination",
    "travel_purpose": "Main reason for travelling", "directness": "Does it go directly to their destination?",
}

# Plain-language explanations reused on the report and dashboard
HOW = {
    "surv": "Start at the top left: at 0 minutes everyone is still waiting (100%). As minutes pass (left to right) people board and the line falls. Where it crosses 50% is the typical wait (P50); where it crosses 10% is the 90th percentile (P90): only 1 passenger in 10 waits longer. The <b>shaded area under the line equals the average wait</b>, so a smaller area means less total waiting. Unfinished waits are included: they count as “still waiting” until 18:00.",
    "ladder": "Each row is one group. The four markers show the 50th, 75th, 90th and 95th percentile wait: half of passengers waited less than the teal dot, 90% less than the red diamond, 95% less than the black tick. A long grey bar means waits there are unpredictable. Rows are sorted by P90.",
    "bands": "Passengers are split into four groups by how long they waited: the fastest half, the next quarter, the next 15% and the slowest 10%. Grey bars show what share of <i>passengers</i> each group is; red bars show what share of all <i>minutes spent waiting</i> they account for.",
    "tailsrc": "We took only the slowest 10% of waits and asked where they happened. Red bars are each stop-direction-and-time-block's share of those waiting minutes; grey bars are its share of all passengers. A long red bar beside a short grey one means that place and time produces far more than its fair share of the long waits.",
    "simcheck": "For every stop-direction and time block, we compare the P90 wait we actually timed (bottom axis) with the P90 our simulation produced from the queue counts and vehicle logs alone (left axis). Dots on the dotted line mean a perfect match.",
    "scen": "Each pair of bars is one option, simulated over the whole observation week. Red = the 90th-percentile wait; teal = the area under the waiting curve, which equals the average wait. Lower is better on both.",
    "hist": "Each bar counts how many timed passengers waited that many minutes. Teal = CNG bus riders, amber = Keke riders; where they overlap the colour is darker. A tall bar on the left means many people boarded almost immediately.",
    "ecdf": "Pick a minute on the bottom axis (the dotted lines mark 5, 10 and 15) and read the height of each line: that is the share of passengers who had already boarded by then. The higher and further left a line is, the better the service.",
    "route": "Each row is one stop and the direction of travel (e.g. <i>NEW MARKET → ACADEMICS</i> means people waiting at New Market to go to the academic area). The bar is the average wait; the red tick shows the wait that the unluckiest 10% of passengers met or exceeded.",
    "boarding": "Each bar is the percentage of passengers at that stop-direction who boarded within the threshold. The thin black whisker is the 95% confidence range: the true value is very likely inside it. Short whiskers mean lots of data.",
    "box": "The box covers the middle half of all waits; the line inside is the typical (median) wait; the whiskers stretch to the usual range. Higher boxes mean longer waits.",
    "hour": "Each point is the typical (median) wait for people who joined the queue during that hour, split by the vehicle type they eventually boarded.",
    "rhythm": "The amber bars (left axis) show how many new passengers arrive at an average stop-direction each hour. The teal line (right axis) shows how many people are standing in the queue at that time. When arrivals rise faster than vehicles take people away, the line climbs.",
    "heat": "Rows are stop-directions, columns are hours of the day. Each square shows the average queue size; darker red = longer queue. Read across a row to see a stop's rush hour; read down a column to compare stops at the same time.",
    "rush": "Each line is one stop-direction. The higher the line at a given hour, the more people were queuing then.",
    "wkend": "Each line shows how many passengers, on average, turned up at a route in each hour. Teal = Monday–Friday, amber = Saturday–Sunday.",
    "friday": "Bars compare the average queue between 13:00 and 14:30 on Fridays (red) with the same window on Monday–Thursday (grey), at New Market and Hall.",
    "vday": "Every coloured bar pointing <b>up</b> is one vehicle arriving: its height is how many people boarded (teal CNG bus, amber Keke). The red bar pointing <b>down</b> underneath is how many people it left behind because it was full. The dotted line is the queue count taken every 10 minutes. Hover a bar to see the vehicle ID and its arrival and departure times.",
    "load": "The bar shows how full vehicles were, on average, when they pulled away: 100% (dotted line) means every seat was taken. Teal = CNG bus (13 seats), amber = Keke (5 seats).",
    "left": "For every vehicle that left, we added the people who boarded and the people left standing. This bar is the left-behind share of that total. A person who misses two vehicles is counted twice, so read it as 'how often vehicles leave people behind', not 'how many different people'.",
    "headway": "Each square is the typical number of minutes between one vehicle arriving and the next, for that stop-direction and hour. Pale = vehicles come often; dark blue = long gaps.",
    "dc": "We divided the passengers arriving in an hour by the seats on vehicles departing in that hour. Green (below 1) = enough seats; red (above 1) = more people than seats, so the queue must grow.",
    "effects": "Each dot is one statistical comparison. The further the dot is from the zero line, the bigger the difference between the two groups. The bar is the 95% confidence range. Teal = statistically significant (p < 0.05); grey = not.",
    "spearman": "Each square shows how strongly two measures rise and fall together across stop-hours. Dark red near +1 = they rise together; dark blue near −1 = one rises as the other falls; white near 0 = no link.",
    "perceived": "For each stop, the amber bar is the average wait students <i>said</i> they usually have (survey), and the teal bar is the average wait we actually <i>timed</i> there.",
    "fleet": "Each bar is one day. The teal part counts the CNG buses and the amber part the Keke that we saw arriving at least once at our stops that day. The red line (right axis) is the share of passengers who boarded within 10 minutes that day.",
    "grid": "Each row is one vehicle (by its ID) and each column a day. A number means we logged that vehicle arriving that many times at our stops; a blank square means it never showed up at our stops that day.",
    "valdays": "For each day we hid that day's data, built the prediction from the other six days, then checked it against every passenger we actually timed on the hidden day. Bars show the average gap between the predicted and the real wait.",
    "calib": "Passengers are grouped by the chance the model gave them of boarding within 10 minutes (bottom axis) and compared with the share who really did (left axis). Points on the dotted line mean the model's percentages can be taken at face value. Bigger dots = more passengers.",
    "survey": "Each bar shows how many surveyed students gave that answer; the label is the share of all respondents.",
}


# ==========================================================================
# Footer (every page)
# ==========================================================================
def footer() -> None:
    names = " · ".join(TEAM["name"].tolist()) if len(TEAM) else "the project team"
    st.markdown(
        f"""<div class='site-footer'>
        <b>OAU Campus Transport Study</b>: field observation of CNG bus and Keke services at Obafemi Awolowo University, Ile-Ife.<br>
        <b>Built and researched by:</b> {names}. All team members carried out the field work and data collection.<br>
        Field records: hand counts by the team, 27–30 September and 27 October 2026. Modelled week: 26 October – 1 November 2026 (estimated to fill stops and hours not observed). Student survey October–November 2026.
        Analysis in Python (pandas, SciPy); dashboard built with Streamlit and Plotly.
        </div>""",
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE 1: REPORT (article)
# ==========================================================================
TOC = [
    ("summary", "In short", False), ("why", "1. Why we did this", False), ("data", "2. Where the data comes from", False),
    ("field", "3. What the field records show", False), ("check", "4. Does the modelled week match the field?", False),
    ("measures", "5. What we measured, and why", False), ("decisions", "6. Decisions and checks", False),
    ("findings", "7. Patterns in the modelled week", False), ("f-shape", "7.1 The shape of waiting", True),
    ("f-where", "7.2 Where the long waits happen", True), ("f-10", "7.3 Boarding within 10 minutes", True),
    ("f-queues", "7.4 When queues build", True), ("f-days", "7.5 Days, Fridays and weekends", True),
    ("f-seats", "7.6 Are there enough seats?", True), ("f-bunch", "7.7 Bunching", True),
    ("f-fleet", "7.8 Vehicles on the road", True), ("f-survey", "7.9 What students told us", True),
    ("f-predict", "7.10 Predicting waits", True), ("f-stats", "7.11 How big are the differences?", True),
    ("cut", "8. How to cut the 90th percentile", False), ("limits", "9. Limitations", False),
    ("conclusion", "10. Conclusion and recommendations", False), ("team", "11. Who did what", False),
]


def report_toc() -> None:
    links = "".join(f"<a class='{'sub' if sub else ''}' href='#{a}'>{t}</a>" for a, t, sub in TOC)
    st.sidebar.markdown("### On this page")
    st.sidebar.markdown(f"<div class='toc'>{links}</div>", unsafe_allow_html=True)


def tag(kind: str) -> str:
    return ("<span class='source-tag field'>Field records</span>" if kind == "field"
            else "<span class='source-tag model'>Modelled week</span>")


def page_report() -> None:
    report_toc()
    W_MAIN = W_ALL[W_ALL["Date"].ge(D.MAIN_WEEK_START)]
    c_all = completed(W_MAIN)
    vm = vehicle_metrics(V_ALL)
    headway, load, dc = D.vehicle_efficiency(Q_ALL, V_ALL)
    load = load.dropna(subset=["Type"])
    dc17 = dc[dc["hour_slot"].le(LAST_HOUR)]
    P = pct_table().iloc[0]
    by_block = pct_table(("Block",)).set_index("Block")
    by_route = pct_table(("stop_direction",))
    rb = pct_table(("stop_direction", "Block"))
    worst = rb[rb["n"] >= 30].sort_values("P90", ascending=False).iloc[0]
    route_ok = c_all.assign(ok=c_all["wait_minutes"].le(10)).groupby("stop_direction")["ok"].mean()
    tail_pct = S.tail_share(c_all["wait_minutes"], 90)
    q75_share = S.tail_share(c_all["wait_minutes"], 75)
    src = tail_sources(c_all)
    top3 = src.head(3)
    per = queue_by_route_hour(Q_ALL)
    peak = per.loc[per["mean_queue"].idxmax()]
    daily_arr = Q_ALL.groupby("Date")["New arrivals (10 min)"].sum()
    over = dc17[dc17["demand_to_capacity_ratio"] > 1]
    fd = fleet_daily(FLEET, Q_ALL, W_ALL)
    scen, sims = scenarios(2)
    base, best = scen.iloc[0], scen.iloc[4]
    dep = departure_waits()
    dep_med = dep.groupby("Type used")[["to_board", "to_depart"]].median()
    dep_p90 = dep.groupby("Type used")[["to_board", "to_depart"]].quantile(.9)
    days = daily_table()
    p_wait_days = permutation_p(days["P90"], days["weekend"])
    p_dem_days = permutation_p(pd.Series(daily_arr.values), pd.Series(pd.to_datetime(daily_arr.index).day_of_week >= 5))
    fsi = field_survival_input(W_F)
    fkm = km_from(fsi)
    f_s10 = fkm.set_index("t").loc[10, "still_waiting"]
    f_p50 = S.km_percentile(fkm, 50)
    f_never = int((~fsi["event"]).sum())
    fvm = vehicle_metrics(V_F)
    f_bscg = fvm[fvm["stop_direction"].eq("BUS STOP -> CAMPUS GATE") & fvm["Type"].eq("CNG bus")]
    f_hall = fvm[fvm["stop_direction"].isin(["CAMPUS GATE -> HALL", "HALL -> CAMPUS GATE"])]
    lb_bad, lb_all = left_behind_check()
    cens_route = W_MAIN[W_MAIN["censored"]].groupby("stop_direction").size().sort_values(ascending=False)
    n_field = len(Q_F) + len(W_F) + len(V_F)
    n_model = len(Q_ALL) + len(W_ALL) + len(V_ALL)
    block_name = {"AM": "morning", "MD": "midday", "PM": "evening"}
    nice = lambda r: r.replace(" -> ", " → ").title()

    # ---- Masthead -------------------------------------------------------
    st.markdown("<div class='kicker'>Field study + modelled week · Campus mobility · OAU Ile-Ife</div>", unsafe_allow_html=True)
    st.markdown("# Waiting for a ride: queues, waits and seats on OAU campus transport")
    st.markdown(
        "<div class='dek'>We went to campus stops in pairs and recorded passengers, queues and vehicles by hand. "
        "To see a whole week at every stop, our workbook fills the hours we could not cover with a model-estimated week. "
        "This report keeps the two apart: what we saw ourselves, what the modelled week suggests, and what would cut the longest waits.</div>",
        unsafe_allow_html=True,
    )
    names = ", ".join(TEAM["name"].tolist())
    st.markdown(f"<div class='byline'>By {names}<br>November 2026 · {n_field:,} field records · {n_model:,} modelled records · {len(SURVEY)} survey responses</div>",
                unsafe_allow_html=True)

    k = st.columns(5)
    k[0].metric("Field: buses leaving full", f"{f_bscg['load'].mean():.0%}", help=f"Bus Stop → Campus Gate CNG buses, {len(f_bscg)} departures recorded by hand")
    k[1].metric("Field: still waiting at 10 min", f"{f_s10:.0%}", help=f"Of {len(fsi)} passengers timed by hand at Bus Stop → Campus Gate")
    k[2].metric("Model: P50 wait", f"{P['P50']:.0f} min", help="Half of passengers in the modelled week waited less than this")
    k[3].metric("Model: P90 wait", f"{P['P90']:.0f} min", help="1 in 10 passengers in the modelled week waited longer than this")
    k[4].metric("Model: tail's share of waiting", f"{tail_pct:.0%}", help="Share of all waiting minutes spent by the slowest 10% of passengers")

    section("In short", "summary", 3)
    with st.container(border=True):
        md(f"""
- {tag('field')} **What we saw ourselves is clear:** at Bus Stop → Campus Gate in the afternoon, buses left **{f_bscg['load'].mean():.0%} full** and left more people behind than they carried. **{f_never} of the {len(fsi)} passengers we timed ({f_never / len(fsi):.0%}) had still not boarded when we stopped.** Buses between Campus Gate and Hall left only about **{f_hall['load'].mean():.0%} full**.
- {tag('model')} **The modelled week gets the same direction** (crowding on the Bus Stop ↔ Campus Gate corridor, spare seats on the Hall routes) and fills in the detail: half of passengers board within **{P['P50']:.0f} minutes**, but 1 in 10 waits more than **{P['P90']:.0f}**, and that slowest 10% accounts for **{tail_pct:.0%} of all waiting time**, concentrated in a few route-hours.
- {tag('model')} **It is more optimistic than reality** at the one place we can check closely: it has far fewer people still waiting at Bus Stop on a Sunday afternoon than we saw. So its waits are probably an **under-estimate**.
- {tag('model')} In a simulation of the modelled week, **moving 2 buses an hour from the quiet Hall routes to the peaks and spacing vehicles evenly cuts P90 from {base['P90']:.0f} to {best['P90']:.0f} minutes and halves total waiting**. The field records point the same way (full buses at Bus Stop, near-empty buses to Hall), but the size of the gain needs a real field week to confirm.
""", )

    # ---- 1. Why ---------------------------------------------------------
    section("1. Why we did this", "why")
    takeaway("Averages hide the problem. We look at the whole spread of waiting time, and especially the slowest trips.")
    md("""
Getting around OAU's campus depends on two kinds of shared vehicle: **CNG buses** (13 passengers) and **Keke** tricycles (5 passengers).
Students complain about long queues, full vehicles driving past and arriving late to lectures, but those complaints are rarely backed by numbers.
A service where most people board at once but a few wait half an hour has a reassuring average and a lot of angry passengers, so we focus on **percentiles** and the **tail**. Our questions:

1. What did we actually see at the stops?
2. Does the modelled week agree with what we saw?
3. What are the 50th, 75th, 90th and 95th percentile waits, and where and when are they worst?
4. Is there enough capacity, and are vehicles where the passengers are?
5. What would reduce the 90th percentile and the total time spent waiting?
""")

    # ---- 2. Data --------------------------------------------------------
    section("2. Where the data comes from", "data")
    takeaway(f"Two kinds of record sit in one workbook: <b>{n_field:,} rows we recorded by hand</b> and <b>{n_model:,} rows from a model-estimated week</b>. We report them separately.")
    md("""
**What we recorded ourselves.** We went to the stops **in pairs of two** and wrote everything down on the spot: one person timing, one recording.
Sheet B2 follows individual passengers from joining the queue to boarding; sheet C logs every vehicle that pulled in (ID, type, arrival and departure time, boarded, got off, left behind, free seats); sheet B1 counts the queue and new arrivals.
Nothing comes from GPS, an app or the operators.

**The modelled week.** One week of hand counting at five stops from 08:00 to 18:00 was beyond our team, so the workbook includes a **model-estimated week (Monday 26 October – Sunday 1 November 2026)** that fills every stop and hour the field work did not cover.
Its header says so (“rows marked ‘Estimated (model)’ only fill times and stops the field work did not cover”), but the shading that marked those rows was lost when the sheet was exported.

**How we told them apart.** Every field row has the recorder's name in **“Entered by”**; every estimated row has only a **“Roster pair (reference)”**. No row has both. The estimated week is also visibly regular: 99% of its queue counts fall exactly on the 10-minute mark, and boarded passengers balance arrivals almost perfectly on every route and day. The team confirmed the split.
""")
    st.markdown(f"""<table class='metrics'>
<tr><th>Sheet</th><th>Field records (by hand)</th><th>When and where (field)</th><th>Modelled week</th></tr>
<tr><td>B1 · Queue counts</td><td>{len(Q_F)}</td><td>Tue 27 Oct, Bus Stop → Campus Gate, 11:12–13:24; one undated afternoon at SUB → Campus Gate</td><td>{len(Q_ALL):,}</td></tr>
<tr><td>B2 · Timed passengers</td><td>{len(W_F)}</td><td>Sun 27 Sept, Bus Stop → Campus Gate, 12:30–15:36</td><td>{len(W_ALL):,}</td></tr>
<tr><td>C · Vehicle arrivals</td><td>{len(V_F)}</td><td>27–30 Sept, six stop-directions, mostly afternoons</td><td>{len(V_ALL):,}</td></tr>
</table>""", unsafe_allow_html=True)
    md("Plus a **student survey** (52 online responses) and two sheets that were planned but left empty, which we rebuilt from the modelled week: **D · fleet availability** (section 7.8) and the **validation log** (section 7.10).")
    st.markdown(
        """<div class='pipeline'>
        <div class='step'><b>1 · Observe</b><span>Pairs at the stops, timing and counting by hand</span></div><div class='arrow'>→</div>
        <div class='step'><b>2 · Fill gaps</b><span>Model-estimated week covers the stops and hours we could not</span></div><div class='arrow'>→</div>
        <div class='step'><b>3 · Separate</b><span>Field rows (named recorder) vs estimated rows (roster only)</span></div><div class='arrow'>→</div>
        <div class='step'><b>4 · Analyse</b><span>Field results; model checked against field; percentiles, survival curves, simulation</span></div><div class='arrow'>→</div>
        <div class='step'><b>5 · Publish</b><span>This report and dashboard</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ---- 3. Field -------------------------------------------------------
    section("3. What the field records show", "field")
    takeaway(f"At Bus Stop → Campus Gate on a Sunday afternoon, a typical tagged passenger waited about <b>{f_p50:.0f} minutes</b>, <b>{f_s10:.0%} were still waiting after 10 minutes</b>, and buses left almost full. Buses to and from Hall ran mostly empty.")
    md(f"""
{tag('field')} These are the records we took by hand. They cover few places and days, so they cannot describe the whole network, but **they are the only direct evidence of what campus transport is like**.

**Timed passengers (Sun 27 Sept, Bus Stop → Campus Gate, CNG bus).** We tagged **{len(fsi)}** passengers between 12:30 and 15:36. **{int(fsi['event'].sum())}** boarded, after 1–12 minutes. **{f_never}** had still not boarded when the session ended; we count them as “waited at least this long” rather than dropping them. Counting them properly, the typical (median) wait was **{f_p50:.0f} minutes** and **{f_s10:.0%}** were still waiting after 10 minutes.
""")
    a, b = st.columns(2)
    with a:
        show(fig_field_waits(W_F))
        explain("Each dot is one passenger: left to right is when they joined the queue, height is how long they waited. Teal dots boarded; red triangles had not boarded when we stopped, so their real wait was at least that long.",
                "Waits stayed under about 12 minutes for people who got on, but the red triangles show a growing group who never did. Dropping them would make the service look much better than it was.")
    with b:
        show(fig_survival({"Field: Bus Stop → Campus Gate, Sun 27 Sept": W_F}, "Share still waiting (field)", "Unfinished waits counted up to the end of the session", colors=[TEAL], xmax=30, field=True))
        f_p90 = S.km_percentile(fkm, 90)
        explain("Start at the top left: at 0 minutes everyone is still waiting. As minutes pass, people board and the line falls. Where it crosses 50% is the typical wait (P50); where it crosses 10% is the P90. Passengers who never boarded count as “still waiting” until the end of the session.",
                f"The curve crosses 50% at about {f_p50:.0f} minutes and is still at {f_s10:.0%} at 10 minutes. It reaches 10% at about {f_p90:.0f} minutes, but by then only a handful of the {len(fsi)} passengers are still being followed, so the field P90 is <b>uncertain and probably longer</b>.")
    md(f"""
**Vehicle log (27–30 Sept, {len(V_F)} arrivals).** CNG buses leaving **Bus Stop → Campus Gate** were **{f_bscg['load'].mean():.0%} full** and left **{f_bscg['left_n'].sum():,.0f}** people behind while carrying **{f_bscg['boarded_n'].sum():,.0f}** (a person missing several buses is counted each time).
Buses between **Campus Gate and Hall** left only about **{f_hall['load'].mean():.0%} full**.

**Queue counts.** On Tue 27 Oct (11:12–13:24) the Bus Stop → Campus Gate queue swung between 0 and 26 people as buses came and went. In one afternoon session at **SUB → Campus Gate** (the date was not written on the sheet) it built from about 20 to 60 people between 14:30 and 16:30.
""")
    a, b = st.columns(2)
    with a:
        show(fig_field_queues(Q_F))
        explain("Each line is one session of hand counts. Height is the number of people queuing at that moment.",
                "The Bus Stop queue empties and refills with each bus. The SUB queue keeps growing through the afternoon: vehicles were not keeping up.")
    with b:
        dates = sorted(V_F["Date"].dropna().dt.date.unique())
        d_sel = st.selectbox("Field day", dates, format_func=lambda d: f"{pd.Timestamp(d):%a %d %b}", key="field_day")
        routes_f = sorted(V_F[V_F["Date"].dt.date.eq(d_sel)]["stop_direction"].dropna().unique())
        r_sel = st.selectbox("Stop → direction", routes_f, key="field_route",
                             index=routes_f.index("BUS STOP -> CAMPUS GATE") if "BUS STOP -> CAMPUS GATE" in routes_f else 0)
        show(fig_vehicle_day(V_F, Q_F, d_sel, r_sel))
        explain(HOW["vday"], "On Sun 27 Sept at Bus Stop → Campus Gate nearly every bus filled up and a red bar (people left behind) sits under almost every arrival: the crowding is real, not a modelling artefact.")

    # ---- 4. Model check -------------------------------------------------
    section("4. Does the modelled week match the field?", "check")
    takeaway("It points the same way (crowding at Bus Stop, spare seats to Hall) but is <b>more optimistic</b> than what we saw, so its waits are likely an under-estimate.")
    md(f"""
Before using the modelled week, we checked it against every place and time where we have field records. {tag('model')} vs {tag('field')}
""")
    st.dataframe(field_model_table(), hide_index=True, width="stretch")
    show(fig_field_vs_model_load(V_F, V_ALL))
    explain("For each stop-direction with field records, teal is how full vehicles left in our hand counts and gold is the modelled week. n is the number of field departures behind each teal bar.",
            "Both show Bus Stop → Campus Gate as the crowded route and the Hall routes as the quiet ones. The model makes Bus Stop a little less crowded and the Hall routes emptier than we saw. The biggest gap is waiting: on a Sunday afternoon the model has almost nobody waiting long at Bus Stop, while we had 40% still waiting after 10 minutes.")
    md("""
**What this means for the rest of the report.** The patterns in sections 7–8 come from the modelled week. Where the field records can check them, the **direction** holds but the **size** is too gentle.
So read the modelled waits as a **lower bound**, treat the exact numbers as illustrations, and treat the statistical tests as describing the model rather than proving facts about campus.
""")

    # ---- 5. Measures ----------------------------------------------------
    section("5. What we measured, and why", "measures")
    takeaway("We judge the service by its slow trips (P90, P95) and by the total waiting it causes (area under the curve), not by the average alone.")
    st.markdown(
        """<table class='metrics'>
<tr><th>Measure</th><th>How we calculated it</th><th>Why we chose it</th></tr>
<tr><td>Wait to board</td><td>Time boarded − time joined, from the two clock times.</td><td>What passengers experience at the kerb.</td></tr>
<tr><td>Wait to depart</td><td>Time the boarded vehicle actually left − time joined.</td><td>A Keke passenger can board and then sit until it fills. This counts that time too, so buses and Keke are compared fairly.</td></tr>
<tr><td>Percentile waits: P50, P75, P90, P95</td><td>The wait that 50/75/90/95% of passengers did not exceed, from a survival (Kaplan–Meier) curve so that unfinished waits are counted, not dropped.</td><td>P50 describes a normal trip; <b>P90 and P95 describe the bad trips</b> behind complaints and late arrivals. Percentiles are not dragged around by a few extreme values.</td></tr>
<tr><td>Area under the waiting curve</td><td>Area under the “share still waiting” curve.</td><td>Equals the average wait; times passenger numbers, it is the <b>total time lost</b>. Shrinking it is the clearest single goal.</td></tr>
<tr><td>Tail share</td><td>Share of all waiting minutes spent by the slowest 10% of passengers.</td><td>Shows how much fixing the tail would achieve.</td></tr>
<tr><td>Queue size, arrivals, headway</td><td>People queuing at each count; new arrivals per hour; minutes between vehicles.</td><td>Demand, supply frequency and regularity.</td></tr>
<tr><td>Load factor, left behind</td><td>Boarded ÷ seats; left behind ÷ (boarded + left behind) at each departure.</td><td>Whether vehicles leave full (short of capacity) or half-empty (capacity in the wrong place).</td></tr>
<tr><td>Demand ÷ seats</td><td>Passengers arriving per hour ÷ seats departing per hour.</td><td>Above 1.0 the queue <i>must</i> grow: the clearest sign of where vehicles are needed.</td></tr>
</table>""",
        unsafe_allow_html=True,
    )

    # ---- 6. Decisions ---------------------------------------------------
    section("6. Decisions and checks", "decisions")
    takeaway("Every choice that could flatter the results was checked, and the results were kept honest even where that made them look worse.")
    md(f"""
- **Field and modelled data are never mixed.** Field results stand on their own (section 3); modelled-week results are labelled as such.
- **Unfinished waits are counted, not dropped.** In the field they are censored at the end of that session; in the modelled week at 18:00. In the modelled week they are only {W_MAIN['censored'].mean():.1%} of passengers, but they **cluster on the worst evening routes** ({", ".join(f"{nice(r)} {n}" for r, n in cens_route.head(2).items())}), so the evening tail there is the least certain part of the results.
- **The 18:00 column is left out of hourly charts.** Observation stopped at 18:00, so that “hour” is a single closing count per route-day and is too thin to interpret.
- **Days, not passengers, are the unit for day-of-week comparisons.** Passengers on the same day share the same weather, events and fleet, so they are not independent. With 5 weekdays and 2 weekend days, we compare days directly.
- **Waits are measured to boarding and to departure.** Keke passengers sit after boarding while the Keke fills, so we report both.
- **No trimming of long waits.** Long waits are the subject, not noise. Capping the top 1% changes the modelled average by under 1%; deleting everything above P90 would hide the passengers this report is about.
- **Impossible counts are flagged, not hidden.** {lb_bad} of {lb_all:,} modelled vehicle records ({lb_bad / lb_all:.1%}) report more people left behind than were in the queue, mostly at Bus Stop ↔ Campus Gate. They are marked in the vehicle charts.
- **Cleaning.** Waits recalculated from clock times; 12-hour times converted to 24-hour; dates read day-first; rows with no destination (or the same stop as origin and destination) excluded; seat capacity 13 (bus) or 5 (Keke) where not written down.
""")

    # ---- 7. Findings (model) --------------------------------------------
    section("7. Patterns in the modelled week", "findings")
    takeaway("These patterns are consistent with the field records in direction. Read the numbers as illustrations and the waits as likely too low.")

    section("7.1 The shape of waiting: a quick start and a long tail", "f-shape", 3)
    md(f"""
{tag('model')} Half of passengers board within **{P['P50']:.0f} minutes (P50)** and three-quarters within **{P['P75']:.0f} (P75)**, but **10% are still waiting after {P['P90']:.0f} (P90)** and 5% after {P['P95']:.0f} (P95). The shaded area, **{P['area']:.1f} minutes**, is the average wait.
**Keke vs CNG bus.** Measured to boarding, a typical Keke rider waits {dep_med.loc['Keke', 'to_board']:.0f} minute against {dep_med.loc['CNG bus', 'to_board']:.0f} for a bus. But a Keke rider then sits while it fills: **measured to departure the typical wait is {dep_med.loc['Keke', 'to_depart']:.0f} minutes for Keke and {dep_med.loc['CNG bus', 'to_depart']:.0f} for buses**, and at P90 the two are close ({dep_p90.loc['Keke', 'to_depart']:.0f} vs {dep_p90.loc['CNG bus', 'to_depart']:.0f}). Keke is faster, but by less than the boarding times suggest.
""")
    show(fig_survival({"All passengers (modelled week)": W_MAIN}, "How long until passengers board?", "Share still waiting after each minute"))
    explain(HOW["surv"], f"Steep at first (many board almost immediately), then a long flat tail that crosses the 10% line at {P['P90']:.0f} minutes. Most of the shaded area sits in that tail.")
    a, b = st.columns(2)
    with a:
        show(fig_bands(c_all))
        explain(HOW["bands"], f"The slowest 10% of passengers spend {tail_pct:.0%} of all waiting minutes, and the slowest quarter {q75_share:.0%}. <b>Fixing the tail is where the gains are.</b>")
    with b:
        show(fig_dep_waits(dep))
        explain("For each vehicle type, the lighter bar is the typical wait to <i>board</i> and the darker bar the typical wait until the vehicle <i>actually leaves</i>. The gap is time spent sitting in a vehicle that is still filling up.",
                f"Keke riders spend about {dep_med.loc['Keke', 'to_depart'] - dep_med.loc['Keke', 'to_board']:.0f} extra minutes seated before leaving; bus riders almost none. Counting it narrows Keke's advantage from {dep_med.loc['CNG bus', 'to_board'] - dep_med.loc['Keke', 'to_board']:.0f} minutes to {dep_med.loc['CNG bus', 'to_depart'] - dep_med.loc['Keke', 'to_depart']:.0f}.")

    section("7.2 Where the long waits happen", "f-where", 3)
    md(f"""
{tag('model')} The worst combination is **{nice(worst['stop_direction'])} in the {block_name[worst['Block']]}** (P50 {worst['P50']:.0f}, P90 {worst['P90']:.0f} minutes). Mornings are worst overall (P90 {by_block.loc['AM', 'P90']:.0f}), midday easiest ({by_block.loc['MD', 'P90']:.0f}).
Three stop-and-time combinations carry **{top3['pax_share'].sum():.0%} of passengers but {top3['share'].sum():.0%} of the slowest waits' minutes**.
""")
    demand = Q_ALL.groupby("stop_direction")["New arrivals (10 min)"].sum() / Q_ALL["Date"].nunique()
    show(fig_route_map(by_route, demand))
    explain("Each arrow is one stop and direction of travel. Colour shows how bad the slowest 10% of waits are; thickness shows how many passengers use it each day. Hover over an arrow for its numbers. Positions are schematic, not a real map.",
            "The thick red arrows cluster around Campus Gate: the Bus Stop ↔ Campus Gate corridor and the New Market routes. The thin teal arrows to Hall and Academics are the reliable, lightly used routes.")
    show(fig_ladder(by_route, "stop_direction", "Percentile waits at each stop and direction", "Sorted by the 90th percentile", target=10))
    worst4 = by_route.sort_values("P90", ascending=False).head(4)
    explain(HOW["ladder"], f"{int((by_route['P90'] <= 10).sum())} of the {len(by_route)} routes keep P90 at or under 10 minutes (dashed line). {nice(worst4.iloc[0]['stop_direction'])} stands out with a P90 of {worst4.iloc[0]['P90']:.0f} minutes, then " + ", ".join(f"{nice(r)} ({p:.0f})" for r, p in zip(worst4['stop_direction'].iloc[1:], worst4['P90'].iloc[1:])) + ".")
    show(fig_tail_sources(c_all))
    explain(HOW["tailsrc"], f"The top three bars carry {top3['pax_share'].sum():.0%} of passengers but {top3['share'].sum():.0%} of the slowest waits' minutes; the top five reach {src.head(5)['share'].sum():.0%}. The tail is a handful of predictable places and times.")

    section("7.3 Will I get on within 10 minutes?", "f-10", 3)
    worst_ok = route_ok.idxmin()
    md(f"{tag('model')} At six of the eleven timed stop-directions more than 85% board within 10 minutes; at **{nice(worst_ok)}** only **{route_ok.loc[worst_ok]:.0%}** do. Use the slider to change the threshold.")
    thr = st.slider("Threshold (minutes)", 5, 20, 10, step=5, key="report_thr")
    show(fig_boarding(c_all, thr))
    explain(HOW["boarding"], "Bars near 100% are reliable routes. The short red-amber bars at the bottom are where a student who leaves only 10 minutes is likely to be late.")

    section("7.4 When the queues build", "f-queues", 3)
    md(f"{tag('model')} The biggest average queue is **{peak['mean_queue']:.0f} people at {nice(peak['stop_direction'])} around {hh(peak['hour_slot'])}**. Hourly charts stop at 17:00 because 18:00 is a single closing count.")
    show(fig_queue_rhythm(Q_ALL))
    explain(HOW["rhythm"], "Demand has a small morning peak, dips around 11:00–12:00, then roughly doubles by 17:00. Queues follow: about 5 people at 08:00–10:00, under 2 at midday, rising again into the evening.")
    show(fig_queue_heat(Q_ALL))
    explain(HOW["heat"], "Two hot zones: Bus Stop → Campus Gate from 15:00 (the trip off campus) and Campus Gate → Bus Stop in the morning (the trip in), plus New Market → Academics in the morning: the same places that produce the waiting tail.")

    section("7.5 Days, Fridays and weekends", "f-days", 3)
    md(f"""
{tag('model')} Comparing days needs care: there is **one week**, so **one Friday** and **two weekend days**. Treating thousands of passengers as independent makes any day look “significant”, but passengers on the same day share the same weather, events and fleet. So we compare **days**:

- **Demand** is clearly lower at weekends: every weekend day had fewer arrivals than every weekday ({daily_arr[pd.to_datetime(daily_arr.index).day_of_week >= 5].max():,} vs at least {daily_arr[pd.to_datetime(daily_arr.index).day_of_week < 5].min():,}); a day-level permutation test gives p = {p_dem_days:.3f}, the smallest possible with 5 vs 2 days.
- **Waits** are not clearly different: weekend P90s ({", ".join(f"{v:.0f}" for v in days[days['weekend']]['P90'])} min) sit inside the weekday range ({days[~days['weekend']]['P90'].min():.0f}–{days[~days['weekend']]['P90'].max():.0f}); day-level p = {p_wait_days:.2f}. Fewer passengers, but also fewer vehicles.
- **Friday at New Market**: the 13:00–14:30 queue averaged about 42 people on the one Friday against under 1 person on every other day. One day cannot tell us whether that is a weekly pattern (for example Friday-prayer travel), an event, or rain. It is a lead to check, not a finding.
""")
    a, b = st.columns(2)
    with a:
        show(fig_daily(days))
        explain("Each bar is one day of the modelled week: the P90 wait that day. With only seven bars, a difference has to be large and consistent to mean anything.",
                f"{days.loc[days['P90'].idxmax(), 'Day']} is the worst day (P90 {days['P90'].max():.0f} min) and {days.loc[days['P90'].idxmin(), 'Day']} the best, but the weekend days sit inside the weekday spread. One week is too short to call a day-of-week effect.")
    with b:
        show(fig_friday(Q_ALL))
        explain(HOW["friday"], "One Friday stands far above the other weekdays at New Market. Because it is a single day, it needs repeat observation on other Fridays before anyone plans around it.")

    section("7.6 Are there enough seats?", "f-seats", 3)
    lf_type = vm.groupby("Type")["load"].mean()
    md(f"""
{tag('model')} On average vehicles leave **{vm['load'].mean():.0%} full** (bus {lf_type.get('CNG bus', np.nan):.0%}, Keke {lf_type.get('Keke', np.nan):.0%}), so there is spare capacity, but in the **wrong place at the wrong time**.
Demand reaches or slightly exceeds the seats on offer in **{len(over)} route-hours**, all on **Bus Stop → Campus Gate (15:00–18:00)** and **New Market → Academics (08:00–10:00)**, with ratios of 1.0–{over['demand_to_capacity_ratio'].max():.1f}. {tag('field')} Our own vehicle log agrees: Bus Stop buses left {f_bscg['load'].mean():.0%} full, Hall buses about {f_hall['load'].mean():.0%}.
""")
    a, b = st.columns(2)
    with a:
        show(fig_load(load))
        explain(HOW["load"], "Bus Stop → Campus Gate is the only route where vehicles routinely leave more than 80% full. Hall ↔ Campus Gate vehicles leave a quarter full or less: the natural place to borrow buses from.")
    with b:
        show(fig_left_behind(load))
        explain(HOW["left"], "The same routes leave the most people behind: Bus Stop → Campus Gate, New Market → Academics/Campus Gate and Campus Gate → Bus Stop.")
    show(fig_dc(dc))
    explain(HOW["dc"], f"Cells above 1.0 sit exactly where the long queues are, but only just above (up to {over['demand_to_capacity_ratio'].max():.2f}): in those hours demand roughly equals the seats on offer, so any bunching or a single missing bus tips the queue into growing.")

    section("7.7 Vehicles arrive bunched together", "f-bunch", 3)
    md(f"{tag('model')} {tag('field')} Even where there are enough seats over the hour, vehicles do not arrive evenly: several within a minute or two of each other, then none for a while. In our field log at Bus Stop → Campus Gate the typical gap was 2 minutes but gaps of 10+ minutes also occur. Passengers arriving just after a bunch face the whole gap.")
    show(fig_headway(headway))
    explain(HOW["headway"], "Busy corridors see a vehicle every 2–3 minutes on average, so their problem is capacity. Quieter routes such as SUB → Hall or Campus Gate → New Market wait 6–10 minutes between vehicles. Section 8 shows how much evening out the gaps would save.")

    section("7.8 How many vehicles were running?", "f-fleet", 3)
    md(f"{tag('model')} The fleet sheet (D) was empty, so we rebuilt it from vehicle IDs seen at the stops in the modelled week: **{fd['fleet_cng'].iat[0]} buses and {fd['fleet_keke'].iat[0]} Keke**, thinning to {int(fd['CNG bus'].iloc[-1])} buses and {int(fd['Keke'].iloc[-1])} Keke on Sunday. Friday had the fewest buses of any weekday ({int(fd['CNG bus'].iloc[4])}) and the worst waits; with one Friday we cannot tell whether that is cause or coincidence.")
    show(fig_fleet_daily(fd))
    explain(HOW["fleet"], "The bars shrink towards the weekend and the red line dips lowest on Friday, the weekday with the fewest buses. At the weekend demand halves too, so service holds up.")
    with st.expander("Which vehicles worked which days (rebuilt sheet D)"):
        show(fig_fleet_grid(FLEET))
        explain(HOW["grid"], "Most vehicles worked most weekdays; the blanks pile up at the weekend. This counts vehicles <i>seen</i>, not vehicles officially in service.")

    section("7.9 What students told us", "f-survey", 3)
    md(f"""
**{len(SURVEY)}** students answered an online survey. **{SURVEY['failed_boarding'].isin(['Often', 'Always']).mean():.0%}** say the first vehicle is *often* or *always* too full to board, the top requests are **shorter waits and more vehicles**, and **{SURVEY['tracking_app'].eq('Yes').mean():.0%}** would use a tracking app.
The survey asked students for their *usual* wait as a range (e.g. “11 to 15 minutes”). That is **not comparable** with our pooled measurements: the respondents chose themselves, they recall their own regular trips (often the crowded peak routes), and the measured median is pulled down by many zero-minute Keke boardings.
So the survey tells us what students **care about and experience**, not a competing measurement of waiting time.
""")
    a, b = st.columns(2)
    with a:
        show(fig_survey_bar(SURVEY, "failed_boarding", "“How often is the first vehicle too full to board?”", SURVEY_ORDERS["failed_boarding"]))
        explain(HOW["survey"], "Most students say <i>often</i> or <i>sometimes</i>; almost none say never. That matches our field log of full buses at Bus Stop.")
    with b:
        show(fig_multi(SURVEY, "desired_improvements", "“What would you most like to see improved?”"))
        explain("Each bar counts students who ticked that improvement; the label is the share of all respondents.",
                "Shorter waits and more vehicles lead: demands about capacity and reliability, not comfort.")
    with st.expander("Reported vs measured waits by boarding stop (not like for like)"):
        show(fig_perceived(COMPARE))
        explain(HOW["perceived"], "Reported waits are higher everywhere, but the comparison is descriptive only: 2–19 self-selected students per stop recalling their usual trips, against all modelled passengers at that stop.")

    section("7.10 Can we predict how long someone will wait?", "f-predict", 3)
    md(f"{tag('model')} Predicting each passenger's wait from the other six days of the modelled week (same stop, direction, time block and vehicle type) is off by **{VAL['abs_error'].mean():.1f} minutes** on average, against **{VAL['baseline_abs_error'].mean():.1f}** for a naive guess. Because the modelled week is itself generated from regular patterns, this mainly shows the method works; a real test needs predictions written down *before* a field session.")
    a, b = st.columns(2)
    with a:
        show(fig_validation_days(VAL))
        explain(HOW["valdays"], "Teal bars are lower than grey on every day: the method beats the naive guess.")
    with b:
        show(fig_calibration(VAL))
        explain(HOW["calib"], "Above about 70% the points sit close to the dotted line; below that the model's percentages are rough.")

    section("7.11 How big are the differences?", "f-stats", 3)
    md("These tests were run on the modelled week with passengers as the unit, so their p-values are **not evidence about campus** (see 7.5). We keep the **effect sizes** as a description of how strongly the modelled week differs between groups (0.1 small, 0.3 medium, 0.5 large).")
    if not TESTS.empty:
        show(fig_effects(TESTS))
        explain(HOW["effects"].replace("Teal = statistically significant (p < 0.05); grey = not.", "Treat these as descriptions of the modelled week, not proof."),
                "The largest effects are the single-Friday New Market surge (0.83) and the morning penalty at New Market and Hall (about 0.5). The Keke vs bus gap is small to medium (0.26), and smaller still once waits are measured to departure.")
    if not SPEARMAN.empty:
        with st.expander("Which measures move together? (Spearman correlations)"):
            show(fig_spearman(SPEARMAN))
            explain(HOW["spearman"], "Queue size is strongly linked with waiting time and with how full vehicles leave.")

    # ---- 8. Cut P90 -----------------------------------------------------
    section("8. How to cut the 90th percentile and the area under the curve", "cut")
    takeaway(f"In a simulation of the modelled week, <b>spacing vehicles evenly and moving 2 buses an hour from the Hall routes to the peaks</b> cuts P90 from {base['P90']:.0f} to {best['P90']:.0f} minutes and halves total waiting, with no new vehicles. The field records support the direction; the size needs a real test.")
    md(f"""
{tag('model')} We built a **queue simulation** that replays the modelled week: passengers arrive at the counted rates (B1) and vehicles arrive at the logged times with their free seats (C), loading first come, first served, plus anyone who walks up while the vehicle is still at the stop.

**How far to trust it.** It reproduces the modelled week's P90 waits within about half a minute across 33 stop-direction × time-block cells, but that check is **partly circular**: the modelled week was itself generated from regular patterns, so matching it shows the simulation is internally consistent, not that it predicts campus. There is not yet enough field data (a full session of arrivals, vehicles and waits at the same stop) to test it against reality.
What makes the direction credible is the field evidence: **full buses and long waits at Bus Stop, near-empty buses on the Hall routes** (sections 3–4).
""")
    obs_rb = pct_table(("stop_direction", "Block"))
    sim_rb = S.percentile_table(sims["As observed"], ["stop_direction", "Block"])
    a, b = st.columns(2)
    with a:
        show(fig_sim_check(obs_rb, sim_rb))
        explain(HOW["simcheck"].replace("the P90 wait we actually timed", "the P90 wait in the modelled week"), "The dots hug the dotted line, so the simulation is consistent with the modelled week, the base we test changes on.")
    with b:
        show(fig_scenarios(scen))
        explain(HOW["scen"], f"Each option helps; combining them helps most: P90 {base['P90']:.0f} → {best['P90']:.0f} minutes, average {base['area']:.1f} → {best['area']:.1f}.")
    md(f"""
**Options tested** (simulated over the modelled week):
1. **Even spacing (same vehicles).** P90 {base['P90']:.0f} → **{scen.iloc[1]['P90']:.0f}** min; area **{scen.iloc[1]['area_change']:.0%}**.
2. **Move 2 buses an hour** from Hall ↔ Campus Gate to the 9 peak route-hours (New Market → Academics and Campus Gate → Bus Stop 08:00–11:00; Bus Stop → Campus Gate 15:00–18:00). Peak P90 {base['peak_P90']:.0f} → **{scen.iloc[2]['peak_P90']:.0f}**; on its own, donor-route P90 rises from {base['donor_P90']:.0f} to {scen.iloc[2]['donor_P90']:.0f}.
3. **Add 2 new buses an hour** in the same cells: P90 → **{scen.iloc[3]['P90']:.0f}**, but it needs extra vehicles and drivers.
4. **Move 2 buses an hour + even spacing.** P90 **{base['P90']:.0f} → {best['P90']:.0f}**, area **{best['area_change']:.0%}**, about **{best['hours_saved_per_day']:.0f} passenger-hours a day** in the modelled week; donor routes end up no worse (P90 {base['donor_P90']:.0f} → {best['donor_P90']:.0f}).
""")
    show(fig_survival({"As modelled": sims["As observed"], "Move 2 buses/hr + even spacing": sims[scen.iloc[4]["scenario"]]},
                      "The waiting curve, before and after", "Simulated over the modelled week", colors=[RED, TEAL]))
    explain(HOW["surv"], "The teal curve falls faster and reaches the 10% line much sooner; its area is about half as large. That smaller area is the waiting time saved.")
    st.markdown("**Try it yourself.**")
    a, b, c = st.columns([1.2, 1, 1])
    extra = a.slider("Extra buses per hour in each peak route-hour", 0, 4, 2, key="sim_extra")
    mode = b.radio("Where do they come from?", ["Moved from Hall ↔ Campus Gate", "Newly added"], key="sim_mode")
    spacing = c.toggle("Space vehicles evenly", value=True, key="sim_spacing")
    if extra == 2 and mode.startswith("Moved") and spacing:
        custom = sims[scen.iloc[4]["scenario"]]
    else:
        custom = custom_scenario(extra, "move" if mode.startswith("Moved") else "add", spacing)
    cp = S.percentile_table(custom).iloc[0]
    kk = st.columns(4)
    kk[0].metric("P50", f"{cp['P50']:.0f} min", f"{cp['P50'] - base['P50']:+.0f}", delta_color="inverse")
    kk[1].metric("P90", f"{cp['P90']:.0f} min", f"{cp['P90'] - base['P90']:+.0f}", delta_color="inverse")
    kk[2].metric("P95", f"{cp['P95']:.0f} min", f"{cp['P95'] - base['P95']:+.0f}", delta_color="inverse")
    kk[3].metric("Average (area)", f"{cp['area']:.1f} min", f"{cp['area'] / base['area'] - 1:+.0%}", delta_color="inverse")
    show(fig_survival({"As modelled": sims["As observed"], "Your option": custom}, "Your option vs the modelled week", colors=[RED, TEAL]))

    # ---- 9. Limitations ------------------------------------------------
    section("9. Limitations: what this study can’t tell you", "limits")
    takeaway("The field records are real but few; the modelled week is complete but estimated, and gentler than reality where we could check it.")
    md(f"""
- **Most records are model-estimated.** {n_model:,} of {n_field + n_model:,} records come from the estimated week. Its patterns agree with the field records in direction but its waits are too optimistic where we can check, and every p-value, correlation and confidence interval computed on it describes the model, not campus.
- **The field records are thin.** One afternoon of timed passengers at one stop (a Sunday), {len(V_F)} vehicle arrivals over four days, and two queue sessions (one undated). They show what happened there and then, not across the network.
- **One week, one Friday.** Day-of-week patterns rest on 7 days; the Friday surge rests on one.
- **Censoring at the end of sessions.** Unfinished waits are only known to be *at least* so long. They cluster on the worst evening routes, so those tails are the least certain; in the field the P90 was longer than our session.
- **Data-entry problems.** {lb_bad} modelled vehicle records report more people left behind than were queuing; left-behind counts also count a person again for each vehicle they miss.
- **Simulation.** Checked only against the modelled week (partly circular); assumes first-come-first-served boarding, unchanged passenger behaviour, and buses that can be dispatched evenly and moved between routes within the hour. Treat its savings as an upper bound for good scheduling, to be tested in the field.
- **Survey.** 52 self-selected students; not comparable with measured waits.
- **Rebuilt sheets.** Fleet availability counts vehicles seen at our stops in the modelled week; the prediction check reuses the same week.
""")

    # ---- 10. Conclusion -------------------------------------------------
    section("10. Conclusion and recommendations", "conclusion")
    takeaway("Crowding is concentrated on the Bus Stop ↔ Campus Gate corridor while buses to Hall run nearly empty, and both our hand counts and the modelled week say so. Move and space the vehicles, and measure properly to confirm the gain.")
    md(f"""
What we saw by hand is consistent and clear: on the **Bus Stop → Campus Gate** run, buses leave essentially full, more people are left behind than carried, and many passengers wait over 10 minutes; buses between **Campus Gate and Hall** leave mostly empty.
The modelled week fills in the rest of the network and shows the same shape: most trips are quick, but a **long tail of slow trips**, concentrated in a few route-hours where demand meets or exceeds the seats on offer and made worse by **bunching**, accounts for a third of all waiting.

**Recommendations**
1. **Run one real field week before deciding anything big.** Use the same sheets but cover the 9 peak route-hours (Bus Stop ↔ Campus Gate, New Market → Academics) on every weekday, keep the “Entered by” column on every row, note the session end time for anyone still waiting, and write predicted waits down *before* each session.
2. **Set a P90 target, not an average target**, e.g. *no stop-direction above a 10-minute P90*, and track P90 and P95 each term.
3. **Space vehicles evenly** at the busiest stops (leave at fixed intervals instead of in convoy). It costs nothing and, in the simulation, cuts P90 by about {base['P90'] - scen.iloc[1]['P90']:.0f} minutes.
4. **Move buses to where the tail is**: about 2 an hour from Hall ↔ Campus Gate to Campus Gate → Bus Stop and New Market → Academics in the morning, and to Bus Stop → Campus Gate from 15:00 to 18:00. New Market → Academics currently has no CNG bus service at all.
5. **Check Fridays at New Market** on several Fridays before planning around them.
6. **Publish expected waits.** {SURVEY['tracking_app'].eq('Yes').mean():.0%} of surveyed students would use a tracking app; even a simple “expected wait” sign reduces the uncertainty that makes waiting feel worse.
""")

    # ---- 11. Team -------------------------------------------------------
    section("11. Who did what", "team")
    md("This was a team effort from planning to publication. **Every member of the team took part in the field work and data collection**, standing at the stops in pairs; the cards below also show what each person led on.")
    team_cards()
    md("The <b>Dashboard</b> page lets you filter every chart and switch between field and modelled data. The <b>Data</b> page lets you download the records.")


def team_cards() -> None:
    if TEAM.empty:
        st.info("Add team members to team.csv.")
        return
    rows = list(TEAM.iterrows())
    for start in range(0, len(rows), 4):
      cols = st.columns(4)
      for col, (_, r) in zip(cols, rows[start:start + 4]):
        initials = "".join(p[0] for p in str(r["name"]).split()[:2]).upper()
        col.markdown(
            f"<div class='team-card'><div class='avatar'>{initials}</div><div class='nm'>{r['name']}</div>"
            f"<div class='rl'>{r['role']}</div><div class='ct'>{r['contributions']}</div></div>",
            unsafe_allow_html=True,
        )


# ==========================================================================
# PAGE 2: DASHBOARD (filters)
# ==========================================================================
def page_dashboard() -> None:
    sb = st.sidebar
    sb.markdown("### Filters")
    sb.caption("Every chart on this page recalculates for your selection.")
    source = sb.radio("Data", ["Modelled week", "Field records only", "Both"], index=0,
                      help="Field records were taken by hand by the team (mostly 27–30 Sept). "
                           "The modelled week (26 Oct – 1 Nov) is estimated to fill stops and hours not observed.")
    if source == "Modelled week":
        q_all, w_all, v_all = Q_ALL, W_ALL, V_ALL
    elif source == "Field records only":
        q_all, w_all, v_all = Q_F, W_F, V_F
    else:
        q_all, w_all, v_all = (pd.concat([Q_ALL, Q_F]), pd.concat([W_ALL, W_F]), pd.concat([V_ALL, V_F]))
    dates = pd.concat([q_all["Date"], w_all["Date"], v_all["Date"]]).dropna()
    dmin, dmax = dates.min().date(), dates.max().date()
    rng = sb.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    d_from, d_to = rng if isinstance(rng, tuple) and len(rng) == 2 else (dmin, dmax)
    days = sb.multiselect("Day of week", DAY_ORDER, default=DAY_ORDER)
    blocks = sb.multiselect("Time block", ["AM", "MD", "PM"], default=["AM", "MD", "PM"],
                            help="AM before 11:00 · MD 11:00–15:00 · PM from 15:00")
    stops = sorted(set(q_all["Stop"].dropna()) | set(w_all["Stop"].dropna()))
    sel_stops = sb.multiselect("Boarding stop", stops, default=stops)
    routes = sorted(r for r in set(q_all["stop_direction"]) | set(w_all["stop_direction"]) | set(v_all["stop_direction"])
                    if isinstance(r, str) and r.split(" -> ")[0] in sel_stops)
    sel_routes = sb.multiselect("Stop → direction", routes, default=routes)
    types = sb.multiselect("Vehicle type", ["CNG bus", "Keke"], default=["CNG bus", "Keke"])
    hours = sb.slider("Hour of day", 8, 18, (8, 18))

    def filt(f: pd.DataFrame, type_col: str | None = None) -> pd.DataFrame:
        m = ((f["Date"].dt.date.between(d_from, d_to) | f["Date"].isna()) & (f["Day"].isin(days) | f["Day"].isna()) & f["Block"].isin(blocks)
             & f["stop_direction"].isin(sel_routes) & f["hour"].between(hours[0], hours[1] + .999))
        if type_col:
            m &= f[type_col].isin(types)
        return f.loc[m.fillna(False)]

    q, w, v = filt(q_all), filt(w_all, "Type used"), filt(v_all, "Type")
    c = completed(w)
    sb.divider()
    sb.caption(f"**In view:** {len(q):,} queue counts · {len(w):,} timed passengers · {len(v):,} vehicle arrivals")

    st.markdown("<div class='kicker'>Interactive dashboard</div>", unsafe_allow_html=True)
    if source != "Field records only":
        st.markdown(tag("model") + " <span style='font-size:.88rem;color:#5b6b70'>Most charts use the model-estimated week. Switch <b>Data</b> in the sidebar to see only what the team recorded by hand.</span>", unsafe_allow_html=True)
    else:
        st.markdown(tag("field") + " <span style='font-size:.88rem;color:#5b6b70'>Only rows recorded by hand. They cover few stops and days, so some charts will be sparse or empty.</span>", unsafe_allow_html=True)
    st.markdown("# Explore the data")
    st.markdown("<div class='dek' style='font-size:1.08rem'>Use the filters on the left to focus on a stop, a day or a time of day. "
                "Every chart has a short guide underneath explaining how to read it.</div>", unsafe_allow_html=True)
    st.write("")

    if c.empty and q.empty and v.empty:
        st.warning("Nothing matches these filters. Widen the selection in the sidebar.")
        return

    vm = vehicle_metrics(v)
    k = st.columns(6)
    k[0].metric("Timed waits", f"{len(c):,}", help=f"{int(w['censored'].sum()):,} unfinished waits excluded")
    kp = S.percentile_table(w).iloc[0] if len(w) else None
    k[1].metric("P50 wait", f"{kp['P50']:.0f} min" if kp is not None else "—")
    k[2].metric("P90 wait", f"{kp['P90']:.0f} min" if kp is not None else "—")
    k[3].metric("Board ≤ 10 min", f"{c['wait_minutes'].le(10).mean():.0%}" if len(c) else "—")
    k[4].metric("Seats filled", f"{vm['load'].mean():.0%}" if len(vm) else "—")
    k[5].metric("Left someone", f"{(vm['left_n'] > 0).mean():.0%}" if len(vm) else "—",
                help="Share of vehicle departures that left at least one person behind")

    tabs = st.tabs(["⏱️ Waiting", "👥 Queues & demand", "🚐 Vehicles & capacity", "🔧 What-if", "📝 Survey", "🧪 Statistics"])
    with tabs[0]:
        if c.empty:
            st.info("No completed waits in this selection.")
        else:
            show(fig_ladder(S.percentile_table(w, ["stop_direction"]), "stop_direction", "Percentile waits at each stop and direction", "Sorted by the 90th percentile", target=10))
            explain(HOW["ladder"], "Rows near the top are the most reliable; long grey bars and red diamonds far to the right mark unpredictable routes.")
            show(fig_survival({"Selection": w}, "How long until passengers board?", "Share still waiting after each minute"))
            explain(HOW["surv"], "Compare the P90 and area in the legend as you change the filters.")
            thr = st.slider("Boarding threshold (minutes)", 1, 30, 10, key="dash_thr")
            show(fig_boarding(c, thr)); explain(HOW["boarding"], "Routes near the bottom are the least reliable for your chosen threshold. Try 5 and 15 minutes to see which routes are fine only if you leave extra time.")
            a, b = st.columns(2)
            with a:
                show(fig_wait_hist(c)); explain(HOW["hist"], "A tall bar at zero with a long tail to the right means service is usually instant but occasionally very slow.")
                show(fig_block_box(c)); explain(HOW["box"], "Compare the height of the boxes to see which part of the day is worst for your selection.")
            with b:
                show(fig_bands(c)); explain(HOW["bands"], "The taller the red bar for the slowest 10%, the more the waiting problem sits in the tail.")
                show(fig_wait_by_hour(c)); explain(HOW["hour"], "Peaks show the hours to avoid; dips show when to travel.")
            with st.expander("Summary table"):
                s = (c.groupby(["stop_direction", "Block"])["wait_minutes"]
                     .agg(n="count", mean="mean", median="median", p90=lambda x: x.quantile(.9)).reset_index()
                     .sort_values("mean", ascending=False))
                st.dataframe(s, hide_index=True, width="stretch", column_config={
                    "stop_direction": "Stop → direction", "n": "Passengers",
                    "mean": st.column_config.ProgressColumn("Average (min)", format="%.1f", min_value=0, max_value=float(s["mean"].max() or 1)),
                    "median": st.column_config.NumberColumn("Typical", format="%.0f"),
                    "p90": st.column_config.NumberColumn("Worst 10%", format="%.0f")})
    with tabs[1]:
        if q.empty:
            st.info("No queue counts in this selection.")
        else:
            show(fig_queue_rhythm(q)); explain(HOW["rhythm"], "If the teal line climbs while the bars stay flat, vehicles are falling behind demand.")
            show(fig_queue_heat(q)); explain(HOW["heat"], "Dark-red squares are the stop-hours to avoid, or where more vehicles are needed.")
            per = queue_by_route_hour(q)
            top = per.groupby("stop_direction")["mean_queue"].max().sort_values(ascending=False).index.tolist()
            pick = st.multiselect("Routes to compare", top, default=top[:4])
            show(fig_rush(q, pick)); explain(HOW["rush"], "The peak of each line is that route's rush hour.")
            a, b = st.columns(2)
            with a:
                show(fig_weekday_weekend(q)); explain(HOW["wkend"], "The gap between the lines is how much quieter weekends are.")
            with b:
                byday = q.groupby("Day")["Queue size"].mean().reindex(DAY_ORDER).dropna().reset_index()
                fig = px.bar(byday, x="Day", y="Queue size", color="Queue size", color_continuous_scale="YlOrRd",
                             labels={"Queue size": "average queue", "Day": ""})
                fig.update_layout(coloraxis_showscale=False)
                show(style(fig, 400, "Average queue by day of week"))
                explain("Each bar is the average queue size across all counts on that weekday.", "Taller bars = busier days for your selection.")
    with tabs[2]:
        if v.empty:
            st.info("No vehicle records in this selection.")
        else:
            headway, load, dc = D.vehicle_efficiency(q, v)
            load = load.dropna(subset=["Type"])
            a, b = st.columns(2)
            with a:
                show(fig_load(load)); explain(HOW["load"], "Bars near the dotted line mean vehicles leave full; short bars mean spare seats.")
            with b:
                show(fig_left_behind(load)); explain(HOW["left"], "Long bars mark where full vehicles regularly leave people standing.")
            show(fig_dc(dc)); explain(HOW["dc"], "Red squares are where demand outstrips supply.")
            show(fig_headway(headway)); explain(HOW["headway"], "Dark squares = long gaps between vehicles.")
            vdates = sorted(v["Date"].dropna().dt.date.unique())
            vroutes = sorted(v["stop_direction"].dropna().unique())
            a, b = st.columns([1, 2])
            dsel = a.selectbox("Date", vdates, index=len(vdates) - 1, format_func=lambda d: f"{pd.Timestamp(d):%a %d %b}", key="vd")
            rsel = b.selectbox("Stop → direction", vroutes, key="vr")
            show(fig_vehicle_day(v, q, dsel, rsel)); explain(HOW["vday"], "Clusters of red bars show when vehicles were leaving full.")
            fl = FLEET[pd.to_datetime(FLEET["Date"]).dt.date.between(d_from, d_to) & FLEET["Type"].isin(types)]
            if len(fl):
                show(fig_fleet_daily(fleet_daily(fl, q, w))); explain(HOW["fleet"], "Days with shorter bars had fewer vehicles working; watch whether the red line dips on those days.")
                show(fig_fleet_grid(fl)); explain(HOW["grid"], "Blank squares show vehicles that never reached our stops that day.")
    with tabs[3]:
        st.caption("Simulated over the full observation week from the queue counts and vehicle logs (does not change with the sidebar filters).")
        scen, sims = scenarios(2)
        base = scen.iloc[0]
        a, b, cc = st.columns([1.2, 1, 1])
        extra = a.slider("Extra buses per hour in each peak route-hour", 0, 4, 2, key="dash_extra")
        mode = b.radio("Where do they come from?", ["Moved from Hall ↔ Campus Gate", "Newly added"], key="dash_mode")
        spacing = cc.toggle("Space vehicles evenly", value=True, key="dash_spacing")
        custom = custom_scenario(extra, "move" if mode.startswith("Moved") else "add", spacing)
        cp = S.percentile_table(custom).iloc[0]
        kk = st.columns(4)
        for col, key, unit in zip(kk, ["P50", "P90", "P95", "area"], ["min"] * 4):
            delta = f"{cp[key] - base[key]:+.0f}" if key != "area" else f"{cp[key] / base[key] - 1:+.0%}"
            col.metric(key if key != "area" else "Area (avg wait)", f"{cp[key]:.0f} min" if key != "area" else f"{cp[key]:.1f} min", delta, delta_color="inverse")
        show(fig_survival({"As observed": sims["As observed"], "Your option": custom}, "Your option vs today", colors=[RED, TEAL]))
        explain(HOW["surv"], "The further left the teal curve crosses the 10% line, the lower the P90.")
        show(fig_scenarios(scen)); explain(HOW["scen"], "The standard options from the report, side by side.")
    with tabs[4]:
        a, b = st.columns([1, 1])
        with a:
            col = st.selectbox("Survey question", list(SURVEY_Q), format_func=SURVEY_Q.get)
            show(fig_survey_bar(SURVEY.dropna(subset=[col]), col, SURVEY_Q[col], SURVEY_ORDERS.get(col)))
            explain(HOW["survey"], "Longer bars are more common answers.")
        with b:
            show(fig_multi(SURVEY, "preference_reasons", "Why students prefer their transport type"))
            explain("Each bar counts students who ticked that reason.", "Comfort comes first, followed by going directly to the destination, shorter waits and faster journeys.")
        show(fig_perceived(COMPARE)); explain(HOW["perceived"], "Amber taller than teal = students feel waits are longer than we measured.")
        comments = SURVEY["other_comments"].dropna()
        comments = comments[comments.str.len() > 3]
        with st.expander(f"Read the students' own comments ({len(comments)})"):
            for t in comments:
                st.markdown(f"> {t}")
    with tabs[5]:
        st.caption("Tests use the full dataset and do not change with the filters.")
        if not TESTS.empty:
            show(fig_effects(TESTS)); explain(HOW["effects"], "Dots far from zero with bars that don't cross zero are the strongest, most certain differences.")
        if not SPEARMAN.empty:
            show(fig_spearman(SPEARMAN)); explain(HOW["spearman"], "Look for dark squares: those pairs of measures are tightly linked.")


# ==========================================================================
# PAGE 3: DATA
# ==========================================================================
def page_data() -> None:
    st.markdown("<div class='kicker'>Open data</div>", unsafe_allow_html=True)
    st.markdown("# The records behind the report")
    md("Every record as analysed (after cleaning). The <b>source</b> column says whether a row was recorded by hand (<i>Field</i>) or comes from the model-estimated week (<i>Estimated</i>). Download any sheet as CSV.")
    choice = st.radio("Dataset", ["B2 · Timed passengers", "B1 · Queue counts", "C · Vehicle arrivals",
                                  "D · Fleet availability", "Validation · Hold-out", "Student survey"], horizontal=True)
    src_pick = st.radio("Rows", ["All", "Field only", "Estimated only"], horizontal=True, key="data_src") \
        if choice[:2] in ("B2", "B1", "C ") else "All"
    allrows = {"B2": F["wait"], "B1": F["queue"], "C ": F["vehicle"]}
    df = {"B2 · Timed passengers": allrows["B2"], "B1 · Queue counts": allrows["B1"], "C · Vehicle arrivals": allrows["C "],
          "D · Fleet availability": FLEET.assign(Date=pd.to_datetime(FLEET["Date"]).dt.strftime("%d/%m/%Y")),
          "Validation · Hold-out": D.validation_sheet(VAL), "Student survey": SURVEY}[choice]
    if choice.startswith("D"):
        st.caption("Rebuilt from sheet C: a vehicle is 'Operating' on a day if it was logged arriving at any observed stop.")
    if choice.startswith("Validation"):
        st.caption("Each passenger's wait predicted from the other six days of the main week, then compared with what was observed.")
    if src_pick != "All" and "source" in df:
        df = df[df["source"].eq("Field" if src_pick == "Field only" else "Estimated")]
    if "source" in df:
        df = df[["source"] + [c for c in df.columns if c != "source"]]
    drop = [c for c in df.columns if c.endswith(" minutes") or c in
            ("Date_original", "time_of_day", "date_day_mismatch", "is_weekend", "valid_route", "joined_minutes",
             "boarded_minutes", "censor_time_available", "end_before_start", "hour")]
    view = df.drop(columns=drop)
    st.caption(f"{len(view):,} rows")
    st.dataframe(view, width="stretch", height=560, hide_index=True)
    st.download_button("⬇️ Download as CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name=choice.split(" · ")[1].lower().replace(" ", "_").replace("-", "") + ".csv", mime="text/csv")


def page_team() -> None:
    st.markdown("<div class='kicker'>Credits</div>", unsafe_allow_html=True)
    st.markdown("# The team")
    md("This study was designed, carried out and analysed by the following people. <b>Every member of the team took part in the field work and data collection</b>, standing at the stops in pairs; each card also shows what that person led on.")
    team_cards()


nav = st.navigation(
    [
        st.Page(page_report, title="Report", icon="📖", default=True),
        st.Page(page_dashboard, title="Dashboard", icon="📊", url_path="dashboard"),
        st.Page(page_data, title="Data", icon="🗂️", url_path="data"),
        st.Page(page_team, title="Team", icon="👥", url_path="team"),
    ],
    position="top",
)
nav.run()
footer()
