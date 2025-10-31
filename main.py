# file: src/water_allocation/fuzzy_allocator.py
from __future__ import annotations

import math
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import csv
import statistics

try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    np = None
    plt = None


# -----------------------------
# Fuzzy Core (minimal, standalone)
# -----------------------------

@dataclass(frozen=True)
class MF:
    """Membership function: callable μ(x). Keep shape params for docs/plots."""
    name: str
    func: Callable[[float], float]
    shape: str
    params: Tuple[float, ...]

    def __call__(self, x: float) -> float:
        return max(0.0, min(1.0, float(self.func(x))))


def trimf(a: float, b: float, c: float, name: str) -> MF:
    """Triangular MF. Why: simple, interpretable."""
    def f(x: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a) if b != a else 0.0
        return (c - x) / (c - b) if c != b else 0.0
    return MF(name=name, func=f, shape="tri", params=(a, b, c))


def trapmf(a: float, b: float, c: float, d: float, name: str) -> MF:
    """Trapezoidal MF. Why: flat tops tolerate measurement noise."""
    def f(x: float) -> float:
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a) if b != a else 0.0
        return (d - x) / (d - c) if d != c else 0.0
    return MF(name=name, func=f, shape="trap", params=(a, b, c, d))


class FuzzyVariable:
    def __init__(self, name: str, universe: Tuple[float, float]):
        self.name = name
        self.universe = universe
        self.sets: Dict[str, MF] = {}

    def add_set(self, label: str, mf: MF) -> "FuzzyVariable":
        self.sets[label] = mf
        return self

    def fuzzify(self, x: float) -> Dict[str, float]:
        return {label: mf(x) for label, mf in self.sets.items()}


@dataclass
class AntecedentAtom:
    var: FuzzyVariable
    label: str

    def degree(self, x: float) -> float:
        return self.var.sets[self.label](x)


@dataclass
class ConsequentAtom:
    var: FuzzyVariable
    label: str


@dataclass
class Rule:
    """Supports AND(min) / OR(max) over antecedents. Multiple consequents."""
    antecedents: List[Tuple[AntecedentAtom, str]]  # (atom, "AND"/"OR")
    consequents: List[ConsequentAtom]
    weight: float = 1.0

    def firing_strength(self, crisp_inputs: Dict[str, float]) -> float:
        # Evaluate chained "AND"/"OR" left-to-right (Mamdani typical, min/max).
        deg = None
        for (atom, op) in self.antecedents:
            val = atom.degree(crisp_inputs[atom.var.name])
            deg = val if deg is None else (min(deg, val) if op == "AND" else max(deg, val))
        return 0.0 if deg is None else max(0.0, min(1.0, deg * self.weight))


class MamdaniSystem:
    def __init__(self):
        self.inputs: Dict[str, FuzzyVariable] = {}
        self.outputs: Dict[str, FuzzyVariable] = {}
        self.rules: List[Rule] = []

    def add_input(self, var: FuzzyVariable) -> "MamdaniSystem":
        self.inputs[var.name] = var
        return self

    def add_output(self, var: FuzzyVariable) -> "MamdaniSystem":
        self.outputs[var.name] = var
        return self

    def add_rule(self, rule: Rule) -> "MamdaniSystem":
        self.rules.append(rule)
        return self

    def infer(self, crisp_inputs: Dict[str, float],
              resolution: int = 501) -> Tuple[Dict[str, float], Dict[str, List[Tuple[float, float]]]]:
        """Return crisp outputs and debug shapes. Why: transparency for SDG governance."""
        # Aggregate fuzzy outputs as clipped unions per output set
        agg_shapes: Dict[str, List[Tuple[float, float]]] = {}
        crisp_outputs: Dict[str, float] = {}

        # Prepare grids
        grids = {
            out_name: _linspace(self.outputs[out_name].universe[0],
                                self.outputs[out_name].universe[1],
                                resolution)
            for out_name in self.outputs
        }
        # Initialize aggregated membership per label
        per_output_label_mu = {
            out_name: {lbl: [0.0]*resolution for lbl in self.outputs[out_name].sets}
            for out_name in self.outputs
        }

        # Fire rules
        for rule in self.rules:
            strength = rule.firing_strength(crisp_inputs)
            if strength <= 0.0:
                continue
            for cons in rule.consequents:
                mf = cons.var.sets[cons.label]
                grid = grids[cons.var.name]
                mu_list = per_output_label_mu[cons.var.name][cons.label]
                # clip MF by firing strength, then union (max)
                for i, x in enumerate(grid):
                    mu_list[i] = max(mu_list[i], min(strength, mf(x)))

        # Aggregate labels to a single shape per output (max across labels)
        for out_name, var in self.outputs.items():
            grid = grids[out_name]
            combined = [0.0]*resolution
            for lbl in var.sets:
                mu = per_output_label_mu[out_name][lbl]
                for i in range(resolution):
                    combined[i] = max(combined[i], mu[i])
            agg_shapes[out_name] = list(zip(grid, combined))
            crisp_outputs[out_name] = _centroid(agg_shapes[out_name])

        return crisp_outputs, agg_shapes


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _centroid(shape: List[Tuple[float, float]]) -> float:
    """Centroid of area with safe fallback."""
    num = 0.0
    den = 0.0
    for x, mu in shape:
        num += x * mu
        den += mu
    if den == 0.0:
        # Safe default: mid-point; Why: no activation shouldn't crash decisions.
        return (shape[0][0] + shape[-1][0]) / 2.0
    return num / den


# -----------------------------
# Problem Model: SDG 6 Water Allocation
# -----------------------------

def build_system() -> MamdaniSystem:
    sys = MamdaniSystem()

    # Inputs
    rainfall = FuzzyVariable("rainfall_mm", (0.0, 400.0)) \
        .add_set("low", trapmf(0, 0, 60, 120, "rf_low")) \
        .add_set("medium", trimf(80, 160, 240, "rf_med")) \
        .add_set("high", trapmf(200, 260, 400, 400, "rf_high"))

    groundwater = FuzzyVariable("groundwater_pct_safe_yield", (0.0, 150.0)) \
        .add_set("depleted", trapmf(0, 0, 25, 50, "gw_dep")) \
        .add_set("normal", trimf(40, 75, 110, "gw_norm")) \
        .add_set("abundant", trapmf(100, 120, 150, 150, "gw_abund"))

    d_agri = FuzzyVariable("demand_agri", (0.0, 100.0)) \
        .add_set("low", trapmf(0, 0, 20, 40, "da_low")) \
        .add_set("medium", trimf(30, 50, 70, "da_med")) \
        .add_set("high", trapmf(60, 80, 100, 100, "da_high"))

    d_ind = FuzzyVariable("demand_ind", (0.0, 100.0)) \
        .add_set("low", trapmf(0, 0, 20, 40, "di_low")) \
        .add_set("medium", trimf(30, 50, 70, "di_med")) \
        .add_set("high", trapmf(60, 80, 100, 100, "di_high"))

    d_hh = FuzzyVariable("demand_hh", (0.0, 100.0)) \
        .add_set("low", trapmf(0, 0, 20, 40, "dh_low")) \
        .add_set("medium", trimf(30, 50, 70, "dh_med")) \
        .add_set("high", trapmf(60, 80, 100, 100, "dh_high"))

    # Outputs (0–100% each, later normalized to 100 total)
    def out_var(name: str) -> FuzzyVariable:
        return FuzzyVariable(name, (0.0, 100.0)) \
            .add_set("very_low", trapmf(0, 0, 5, 15, "vl")) \
            .add_set("low", trimf(10, 20, 35, "l")) \
            .add_set("medium", trimf(30, 45, 60, "m")) \
            .add_set("high", trimf(55, 70, 85, "h")) \
            .add_set("very_high", trapmf(80, 90, 100, 100, "vh"))

    a_alloc = out_var("alloc_agri")
    i_alloc = out_var("alloc_ind")
    h_alloc = out_var("alloc_hh")

    # Register vars
    for v in (rainfall, groundwater, d_agri, d_ind, d_hh):
        sys.add_input(v)
    for v in (a_alloc, i_alloc, h_alloc):
        sys.add_output(v)

    A = lambda var, lbl: (AntecedentAtom(var, lbl), "AND")
    O = lambda var, lbl: (AntecedentAtom(var, lbl), "OR")
    C = lambda var, lbl: ConsequentAtom(var, lbl)

    rules: List[Rule] = []

    def R(ants: List[Tuple[AntecedentAtom, str]], cons: List[ConsequentAtom], w: float = 1.0):
        rules.append(Rule(ants, cons, w))

    # Scarcity priority to households; agriculture reduced; industry moderate.
    R([A(rainfall, "low"), A(groundwater, "depleted")],
      [C(h_alloc, "very_high"), C(i_alloc, "medium"), C(a_alloc, "very_low")])

    # Low rain, normal GW; if HH demand high → HH high, Agri low, Ind medium.
    R([A(rainfall, "low"), A(groundwater, "normal"), A(d_hh, "high")],
      [C(h_alloc, "very_high"), C(a_alloc, "low"), C(i_alloc, "medium")])

    # Abundance boosts agriculture when agri demand high.
    R([A(rainfall, "high"), A(groundwater, "abundant"), A(d_agri, "high")],
      [C(a_alloc, "high"), C(i_alloc, "medium"), C(h_alloc, "medium")])

    # Industry surge under normal water
    R([A(d_ind, "high"), A(groundwater, "normal"), A(rainfall, "medium")],
      [C(i_alloc, "high"), C(a_alloc, "medium"), C(h_alloc, "medium")])

    # Households protected when HH demand high irrespective of others
    R([O(d_hh, "high")],
      [C(h_alloc, "high")], w=0.9)

    # Agriculture protected in monsoon even if HH medium
    R([A(rainfall, "high"), O(d_hh, "medium")],
      [C(a_alloc, "high"), C(h_alloc, "medium"), C(i_alloc, "medium")])

    # Industry curtailed if GW depleted and Ind demand low
    R([A(groundwater, "depleted"), A(d_ind, "low")],
      [C(i_alloc, "low")])

    # If Agri demand low → shift to HH/Ind
    R([A(d_agri, "low")],
      [C(a_alloc, "low"), C(h_alloc, "medium"), C(i_alloc, "medium")])

    # Balanced conditions → balanced allocations
    R([A(rainfall, "medium"), A(groundwater, "normal"),
       A(d_agri, "medium"), A(d_ind, "medium"), A(d_hh, "medium")],
      [C(a_alloc, "medium"), C(i_alloc, "medium"), C(h_alloc, "medium")])

    # Drought but Ind high demand → protect HH, keep Ind medium, cut Agri
    R([A(rainfall, "low"), A(groundwater, "depleted"), A(d_ind, "high")],
      [C(h_alloc, "high"), C(i_alloc, "medium"), C(a_alloc, "low")])

    # Plenty water & low HH → push to productive uses
    R([A(rainfall, "high"), A(groundwater, "abundant"), A(d_hh, "low")],
      [C(h_alloc, "low"), C(i_alloc, "high"), C(a_alloc, "high")])

    # Normal water & Agri high → Agri high, others medium
    R([A(groundwater, "normal"), A(d_agri, "high")],
      [C(a_alloc, "high"), C(i_alloc, "medium"), C(h_alloc, "medium")])
    
    # Extreme drought & ALL demands high -> prioritize Household > Industry > Agriculture
    R([A(rainfall, "low"), A(groundwater, "depleted"),
       A(d_agri, "high"), A(d_ind, "high"), A(d_hh, "high")],
      [C(h_alloc, "very_high"), C(i_alloc, "medium"), C(a_alloc, "low")], w=1.0)

    # All abundant & ALL demands low -> balanced medium allocation to promote flexibility
    R([A(rainfall, "high"), A(groundwater, "abundant"),
       A(d_agri, "low"), A(d_ind, "low"), A(d_hh, "low")],
      [C(a_alloc, "medium"), C(i_alloc, "medium"), C(h_alloc, "medium")], w=1.0)

    # Add all rules
    for rule in rules:
        sys.add_rule(rule)

    return sys


# -----------------------------
# Allocation API
# -----------------------------

@dataclass
class AllocationResult:
    agri: float
    industry: float
    household: float
    raw_agri: float
    raw_industry: float
    raw_household: float
    reason: Dict[str, List[Tuple[float, float]]]  # aggregated shapes per output


def allocate(rainfall_mm: float,
             groundwater_pct_safe_yield: float,
             demand_agri: float,
             demand_ind: float,
             demand_hh: float,
             min_floor: Tuple[float, float, float] = (10.0, 10.0, 20.0),
             max_cap: Tuple[float, float, float] = (70.0, 60.0, 70.0)
             ) -> AllocationResult:
    """
    Compute allocations. Floors/caps encode fairness & rights to water (HH floor higher).
    """
    sys = build_system()
    crisp_in = dict(
        rainfall_mm=rainfall_mm,
        groundwater_pct_safe_yield=groundwater_pct_safe_yield,
        demand_agri=demand_agri,
        demand_ind=demand_ind,
        demand_hh=demand_hh,
    )
    raw, shapes = sys.infer(crisp_in)

    ra = float(raw["alloc_agri"])
    ri = float(raw["alloc_ind"])
    rh = float(raw["alloc_hh"])

    # Normalize to sum 100, then enforce floors/caps with iterative projection.
    norm = _normalize_to_100((ra, ri, rh))
    clamped = _enforce_fairness(norm, min_floor, max_cap)
    return AllocationResult(
        agri=clamped[0], industry=clamped[1], household=clamped[2],
        raw_agri=ra, raw_industry=ri, raw_household=rh,
        reason=shapes
    )


def _normalize_to_100(vals: Tuple[float, float, float]) -> Tuple[float, float, float]:
    s = sum(vals)
    if s <= 0.0:
        return (33.34, 33.33, 33.33)
    return tuple(v * 100.0 / s for v in vals)  # type: ignore[return-value]


def _enforce_fairness(vals: Tuple[float, float, float],
                      floors: Tuple[float, float, float],
                      caps: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Project onto simplex with min/max bounds. Why: protect rights & avoid over-allocation.
    """
    v = [max(f, min(c, x)) for x, f, c in zip(vals, floors, caps)]
    # Re-normalize while respecting bounds (simple water-filling)
    total = sum(v)
    if math.isclose(total, 100.0, rel_tol=1e-6, abs_tol=1e-6):
        return tuple(v)  # type: ignore[return-value]

    # Distribute delta
    delta = 100.0 - total
    free = [i for i in range(3) if (delta > 0 and v[i] < caps[i]) or (delta < 0 and v[i] > floors[i])]
    if not free:
        # If no freedom, accept closest feasible
        scale = 100.0 / total
        return tuple(min(caps[i], max(floors[i], v[i] * scale)) for i in range(3))  # type: ignore[return-value]

    while abs(delta) > 1e-6 and free:
        share = delta / len(free)
        new_free = []
        for i in free:
            if delta > 0:
                room = caps[i] - v[i]
                inc = min(room, share)
                v[i] += inc
                delta -= inc
            else:
                room = v[i] - floors[i]
                dec = min(room, -share)
                v[i] -= dec
                delta += dec
            # keep if still has room
            if (delta > 0 and v[i] < caps[i]) or (delta < 0 and v[i] > floors[i]):
                new_free.append(i)
        if new_free == free:
            break
        free = new_free

    # Final small correction
    correction = 100.0 - sum(v)
    if free:
        v[free[0]] += correction
    else:
        v[0] += correction
    return tuple(v)  # type: ignore[return-value]


# -----------------------------
# CSV ingestion (optional)
# -----------------------------

def load_kaggle_like_csv(path: str,
                         rainfall_col: str = "rainfall_mm",
                         groundwater_col: str = "groundwater_pct_safe_yield",
                         agri_col: str = "demand_agri",
                         ind_col: str = "demand_ind",
                         hh_col: str = "demand_hh") -> List[Dict[str, float]]:
    """
    Expect columns named as above (rename if needed). Scales demand columns to 0–100 if not already.
    """
    rows: List[Dict[str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = [r for r in reader]

    # Try to min-max scale demand fields to 0-100 if outside range
    def col_float(vals, key):
        out = []
        for r in vals:
            try:
                out.append(float(r[key]))
            except Exception:
                out.append(float("nan"))
        return out

    for key in [agri_col, ind_col, hh_col]:
        arr = [float(r.get(key, "nan")) for r in raw_rows]
        clean = [x for x in arr if not math.isnan(x)]
        if not clean:
            continue
        mn, mx = min(clean), max(clean)
        if mn < 0 or mx > 100 or (mx - mn) > 100:
            # scale
            for r in raw_rows:
                try:
                    val = float(r.get(key, "nan"))
                    r[key] = 0.0 if math.isnan(val) else 0.0 if mx == mn else (val - mn) * 100.0 / (mx - mn)
                except Exception:
                    r[key] = 0.0

    for r in raw_rows:
        try:
            rows.append(dict(
                rainfall_mm=float(r[rainfall_col]),
                groundwater_pct_safe_yield=float(r[groundwater_col]),
                demand_agri=float(r[agri_col]),
                demand_ind=float(r[ind_col]),
                demand_hh=float(r[hh_col]),
            ))
        except Exception:
            # skip bad row
            continue
    return rows


# -----------------------------
# Visualization (optional)
# -----------------------------

def plot_memberships(sys: MamdaniSystem):
    if plt is None or np is None:
        return
    plt.figure()
    plt.title("Rainfall membership")
    x = np.linspace(*sys.inputs["rainfall_mm"].universe, 500)
    for lbl, mf in sys.inputs["rainfall_mm"].sets.items():
        y = [mf(xx) for xx in x]
        plt.plot(x, y, label=lbl)
    plt.legend()
    plt.xlabel("mm/month")
    plt.ylabel("μ")

    plt.figure()
    plt.title("Allocation (Household) membership")
    x = np.linspace(*sys.outputs["alloc_hh"].universe, 500)
    for lbl, mf in sys.outputs["alloc_hh"].sets.items():
        y = [mf(xx) for xx in x]
        plt.plot(x, y, label=lbl)
    plt.legend()
    plt.xlabel("%")
    plt.ylabel("μ")


def plot_allocation_pie(alloc: AllocationResult):
    if plt is None:
        return
    plt.figure()
    plt.title("Water Allocation (%)")
    data = [alloc.agri, alloc.industry, alloc.household]
    labels = ["Agriculture", "Industry", "Household"]
    plt.pie(data, labels=labels, autopct="%1.1f%%")

# -----------------------------
# Visualization (agg_shapes)
# -----------------------------
def plot_aggregated_outputs(agg_shapes: Dict[str, List[Tuple[float, float]]]):

    if plt is None:
        return
    for out_name, shape in agg_shapes.items():
        if not shape:
            continue
        x, y = zip(*shape)
        plt.figure()
        plt.plot(x, y, label=f"{out_name} aggregated μ(x)")
        plt.fill_between(x, y, alpha=0.2)
        plt.title(f"Aggregated Output Membership - {out_name}")
        plt.xlabel("Output value")
        plt.ylabel("Membership degree")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()

# -----------------------------
# CLI & Demo
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Sustainable Water Allocation Advisor (Mamdani FIS)")
    parser.add_argument("--rain", type=float, default=90.0, help="Rainfall (mm/month)")
    parser.add_argument("--gw", type=float, default=55.0, help="Groundwater (% of safe yield)")
    parser.add_argument("--da", type=float, default=60.0, help="Agri demand index (0-100)")
    parser.add_argument("--di", type=float, default=40.0, help="Industry demand index (0-100)")
    parser.add_argument("--dh", type=float, default=70.0, help="Household demand index (0-100)")
    parser.add_argument("--csv", type=str, default="", help="Batch mode: path to CSV")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")
    args = parser.parse_args()

    if args.csv:
        rows = load_kaggle_like_csv(args.csv)
        print("date,agri,industry,household,raw_agri,raw_industry,raw_household")
        for r in rows:
            res = allocate(r["rainfall_mm"], r["groundwater_pct_safe_yield"],
                           r["demand_agri"], r["demand_ind"], r["demand_hh"])
            print(f"{r.get('date','')},{res.agri:.2f},{res.industry:.2f},{res.household:.2f},"
                  f"{res.raw_agri:.2f},{res.raw_industry:.2f},{res.raw_household:.2f}")
        return

    res = allocate(args.rain, args.gw, args.da, args.di, args.dh)
    print("=== Allocation (normalized to 100%) ===")
    print(f"Agriculture: {res.agri:.2f}% (raw {res.raw_agri:.2f})")
    print(f"Industry   : {res.industry:.2f}% (raw {res.raw_industry:.2f})")
    print(f"Household  : {res.household:.2f}% (raw {res.raw_household:.2f})")

    if not args.no_plots:
        sys = build_system()
        plot_memberships(sys)
        plot_allocation_pie(res)
        if plt is not None:
            plt.show()

# --- New: Raw vs Adjusted Allocations bar chart ---
            if plt is not None:
                        plt.figure()
                        plt.title("Raw vs Fairness-Adjusted Allocations")
                        labels = ["Agriculture", "Industry", "Household"]
                        raw_vals = [res.raw_agri, res.raw_industry, res.raw_household]
                        adj_vals = [res.agri, res.industry, res.household]
                        x = range(len(labels))
                        width = 0.35
                        plt.bar([i - width/2 for i in x], raw_vals, width=width, label="Raw")
                        plt.bar([i + width/2 for i in x], adj_vals, width=width, label="Adjusted")
                        plt.xticks(x, labels)
                        plt.ylabel("Allocation (%)")
                        plt.ylim(0, 100)
                        plt.legend()
                        plt.grid(axis="y", linestyle=":", linewidth=0.5)
            

# --- New: show aggregated output membership shapes (transparency) ---
            plot_aggregated_outputs(res.reason)
            
            if plt is not None:
                plt.show()
# -----------------------------
# Minimal tests (run: python fuzzy_allocator.py --no-plots)
# -----------------------------

def _self_test():
    # Drought: expect HH high floor, Agri low
    res = allocate(10, 10, 70, 30, 90)
    assert res.household >= 50 and res.agri <= 25

    # Abundance & Agri high: Agri should be higher than HH
    res = allocate(350, 140, 90, 50, 30)
    assert res.agri > res.household

    # Floors respected
    res = allocate(0, 0, 0, 0, 0, min_floor=(10, 10, 30))
    assert res.household >= 30 - 1e-6

    # Normalization
    s = res.agri + res.industry + res.household
    assert abs(s - 100.0) < 1e-6


if __name__ == "__main__":
    _self_test()
    main()
