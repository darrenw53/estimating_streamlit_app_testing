# logic.py
import csv
import os
import math

# --- Constants ---
DENSITY_FACTOR_FOR_CALCULATION = 0.282 # For plates, lbs/cubic inch
PERCENTAGE_ADD_FOR_GROSS_WEIGHT = 0.20 # 20%
FEEDRATE_TABLE_IPM = {
    0.188: 200.0, 0.25: 160.0, 0.3125: 135.0, 0.375: 125.0, 0.5: 95.0,
    0.625: 75.0,  0.75: 65.0,  0.875: 50.0,  1.0: 45.0,   1.125: 42.0,
    1.25: 40.0,   1.375: 35.0, 1.5: 30.0,    1.625: 26.0,  1.75: 24.0,
    2.0: 10.0,    2.25: 10.0,  2.5: 9.0,     2.75: 8.5,   3.0: 8.0,
    3.25: 7.5,   3.5: 7.0,    3.75: 7.0,    4.0: 7.0,    5.0: 6.0,
    6.0: 6.0,     7.0: 6.0
}
BEND_TIME_PER_COMPLEXITY_MINUTES = {
    "Simple": 15, "Moderate": 30, "Complex": 60, "N/A": 0
}
# Default to the CSV shipped with this project (works locally + on Streamlit Cloud)
AISC_CSV_FILENAME = os.path.join(
    os.path.dirname(__file__),
    "aisc-shapes-database-v15.0.xlsx - Database v15.0.csv",
)
STRUCTURAL_CUTTING_RATE_SQ_IN_PER_MIN = 2.0
STRUCTURAL_TRAVEL_TIME_PER_PIECE_MIN = 2.0
MITER_CUT_MULTIPLIER = 2.0
MM_PER_INCH = 25.4
SQMM_PER_SQINCH = MM_PER_INCH * MM_PER_INCH
AREA_CONVERSION_THRESHOLD_SQIN = 500
PLATE_BURN_MACHINE_THICKNESS_THRESHOLD = 0.625
DRILL_SFM = 60.0
DRILL_IPR = 0.005
PLATE_DRILL_TRAVEL_TIME_PER_HOLE_MIN = 15.0 / 60.0

# --- Welding Data & Multipliers ---
WELD_DATA = {
    "1/8\" (0.125)": {"lbs_per_ft": 0.06, "in_per_hr": 144},
    "3/16\" (0.188)": {"lbs_per_ft": 0.13, "in_per_hr": 120},
    "1/4\" (0.250)": {"lbs_per_ft": 0.18, "in_per_hr": 120},
    "5/16\" (0.312)": {"lbs_per_ft": 0.29, "in_per_hr": 96},
    "3/8\" (0.380)": {"lbs_per_ft": 0.42, "in_per_hr": 96},
    "1/2\" (0.500)": {"lbs_per_ft": 0.77, "in_per_hr": 60},
    "5/8\" (0.625)": {"lbs_per_ft": 1.15, "in_per_hr": 52},
    "3/4\" (0.750)": {"lbs_per_ft": 1.63, "in_per_hr": 60},
    "7/8\" (0.880)": {"lbs_per_ft": 2.25, "in_per_hr": 28},
    "1\" (1.000)": {"lbs_per_ft": 2.92, "in_per_hr": 20}
}
WELD_SIZE_OPTIONS = list(WELD_DATA.keys())
PRE_HEAT_MULTIPLIER = 1.5
CJP_MULTIPLIER = 2.0

# --- Global Variables for AISC Data ---
AISC_TYPES_TO_LABELS_MAP = None
AISC_LABEL_TO_PROPERTIES_MAP = None
aisc_data_load_attempted = False

# --- Data for Form Dropdowns ---
MATERIALS_LIST = ["A36", "A572", "A514", "A633", "A656"]
THICKNESS_LIST = sorted(list(FEEDRATE_TABLE_IPM.keys()))
BEND_COMPLEXITY_OPTIONS = ["Simple", "Moderate", "Complex"]

def load_aisc_database(filepath=AISC_CSV_FILENAME):
    global AISC_TYPES_TO_LABELS_MAP, AISC_LABEL_TO_PROPERTIES_MAP, aisc_data_load_attempted
    if aisc_data_load_attempted: return
    aisc_data_load_attempted = True
    AISC_TYPES_TO_LABELS_MAP, AISC_LABEL_TO_PROPERTIES_MAP = {}, {}
    if not os.path.exists(filepath):
        print(f"Error: AISC Database file not found at '{filepath}'")
        return
    temp_types_to_labels, temp_label_to_properties = {}, {}
    try:
        with open(filepath, mode='r', newline='', encoding='latin1') as csvfile:
            reader = csv.DictReader(csvfile)
            required_cols = ['Type', 'EDI_Std_Nomenclature', 'W', 'A']
            if not all(col in reader.fieldnames for col in required_cols):
                 print(f"Error: Required columns ('Type', 'EDI_Std_Nomenclature', 'W', 'A') not found in '{filepath}'.")
                 return
            for row in reader:
                shape_type, label = row.get('Type','').strip(), row.get('EDI_Std_Nomenclature','').strip()
                weight_str, area_str = row.get('W','').strip(), row.get('A','').strip()
                if shape_type and label:
                    if shape_type not in temp_types_to_labels: temp_types_to_labels[shape_type] = []
                    if label not in temp_types_to_labels[shape_type]: temp_types_to_labels[shape_type].append(label)
                    if label not in temp_label_to_properties:
                        properties = {k.strip(): v.strip() for k, v in row.items()}
                        try: properties['W_float'] = float(weight_str) if weight_str else 0.0
                        except: properties['W_float'] = 0.0
                        try:
                            area_val = float(area_str) if area_str else 0.0
                            if area_val > AREA_CONVERSION_THRESHOLD_SQIN: area_val = area_val / SQMM_PER_SQINCH
                            properties['A_float'] = area_val
                        except: properties['A_float'] = 0.0
                        temp_label_to_properties[label] = properties
        for st_key in temp_types_to_labels:
            def natural_sort_key(s):
                parts, c = [], "";
                for char in s:
                    if char.isdigit() != c.isdigit() and c: parts.append(int(c) if c.isdigit() else c.lower()); c = ""
                    c += char
                if c: parts.append(int(c) if c.isdigit() else c.lower())
                return parts
            try: temp_types_to_labels[st_key].sort(key=natural_sort_key)
            except: temp_types_to_labels[st_key].sort()
        AISC_TYPES_TO_LABELS_MAP, AISC_LABEL_TO_PROPERTIES_MAP = temp_types_to_labels, temp_label_to_properties
        if AISC_TYPES_TO_LABELS_MAP: print(f"Successfully processed AISC data from '{filepath}'")
    except Exception as e: print(f"Error processing AISC file '{filepath}': {e}")


# --- Calculation Functions ---
def calculate_structural_end_perimeter(props: dict) -> float:
    """
    Approximate the perimeter around a structural shape cross-section at ONE end (inches).

    Notes:
    - For round shapes (Pipe / HSS round), uses pi * OD.
    - For HSS/Tube rectangular, uses 2*(B + Ht) when available.
    - For other shapes (W, C, L, etc.), falls back to 2*(bf + d) or 2*(b + d) as an envelope approximation.

    The caller can multiply by 2 for 'both ends'.
    """
    import math

    def f(key: str) -> float:
        try:
            v = props.get(key, 0.0)
            if v is None or v == "":
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    od = f("OD")
    if od > 0:
        return math.pi * od

    # HSS/Tube rectangles commonly have B and Ht
    B = f("B") or f("bf") or f("b")
    Ht = f("Ht") or f("d")
    if B > 0 and Ht > 0:
        return 2.0 * (B + Ht)

    # Fallback
    w = f("bf") or f("B") or f("b")
    d = f("d") or f("Ht")
    if w > 0 and d > 0:
        return 2.0 * (w + d)

    return 0.0
def get_plate_burn_machine_type(t): return "Laser" if t <= PLATE_BURN_MACHINE_THICKNESS_THRESHOLD else "Kinetic"
def calculate_plate_perimeter(w, l): return (w * 2) + (l * 2) if w > 0 and l > 0 else 0.0
def get_feedrate_for_thickness(t, ft): return ft.get(t, 1.0)
def calculate_burning_time(li, fr): return li / fr if li > 0 and fr and fr > 0 else 0.0
def calculate_drilling_time_per_hole(dia, d, sfm, ipr, travel):
    if not all(isinstance(v,(float,int)) and v > 0 for v in [dia, d, sfm, ipr]): return 0.0
    try:
        rpm = (sfm * 12) / (math.pi * dia)
        feed = rpm * ipr
        machining_time = d / feed if feed > 0 else float('inf')
        return machining_time + travel if machining_time != float('inf') else travel
    except: return travel
def process_plate_drilling_data(form, thickness):
    total_time, summary_list = 0.0, []
    for i in range(1, 4):
        try:
            dia, qty = float(form.get(f'hole_dia_{i}',0)), int(form.get(f'hole_qty_{i}',0))
            if dia > 0 and qty > 0:
                time_per = calculate_drilling_time_per_hole(dia, thickness, DRILL_SFM, DRILL_IPR, PLATE_DRILL_TRAVEL_TIME_PER_HOLE_MIN)
                if time_per < float('inf'): total_time += time_per * qty; summary_list.append(f"{qty}x ø{dia}\"")
        except: pass
    return round(total_time,2), "; ".join(summary_list) if summary_list else "None"
def calculate_bend_time(nb, bc, tm): return float(nb * tm.get(bc, 0)) if nb > 0 else 0.0
def calculate_plate_net_weight(t,w,l,df): return (t*w*l)*df if all(v>0 for v in [t,w,l,df]) else 0.0
def calculate_structural_piece_weight(wpf,lin): return (wpf/12.0)*lin if wpf>0 and lin>0 else 0.0
def calculate_gross_weight(nw,pi): return nw*(1+pi) if nw>=0 else 0.0
def calculate_fit_time(pw):
    if not isinstance(pw, (float, int)) or pw < 0: return 0
    if pw <= 50: return 15
    elif pw <= 150: return 30
    elif pw <= 500: return 45
    else: return 60
def calculate_structural_cutting_time(area,rate,travel):
    if not isinstance(area,(float,int)) or area<=0: return 0.0
    if not isinstance(rate,(float,int)) or rate<=0: return travel
    return round((area/rate)+travel,2)
def calculate_yield_for_stock_size(cuts_orig, stock_len):
    if not cuts_orig or stock_len<=0: return 0, 0
    cuts = sorted([c for c in cuts_orig if 0<c<=stock_len], reverse=True)
    if not cuts: return 0,0
    bars, total_waste, cuts_to_make = 0, 0.0, list(cuts)
    while cuts_to_make:
        bars += 1; rem_len = float(stock_len)
        remaining_cuts = []
        for cut in cuts_to_make:
            if cut <= rem_len: rem_len -= cut
            else: remaining_cuts.append(cut)
        cuts_to_make = remaining_cuts
        total_waste += rem_len
    return bars, total_waste


# ============================================================
# Structural nesting (1D cutting stock) — robust multi-length mix
# ============================================================

def _expand_cuts_from_rows(cuts_with_qty):
    """Expand [(length, qty), ...] into a flat list of lengths (floats)."""
    expanded = []
    if not cuts_with_qty:
        return expanded
    for item in cuts_with_qty:
        try:
            if isinstance(item, dict):
                length = float(item.get("length", 0.0) or 0.0)
                qty = int(item.get("qty", 0) or 0)
            else:
                length = float(item[0])
                qty = int(item[1])
            if length > 0 and qty > 0:
                expanded.extend([length] * qty)
        except Exception:
            continue
    return expanded


def _pack_one_bar_best_fit(cuts_desc, stock_len, kerf=0.0, end_trim=0.0):
    """Pack as many cuts as possible into one bar using best-fit on a descending list.

    Returns:
      used_cuts (list[float]), remaining_cuts (list[float]), waste (float)
    """
    kerf = float(kerf or 0.0)
    end_trim = float(end_trim or 0.0)
    capacity = float(stock_len) - end_trim
    if capacity <= 0:
        return [], list(cuts_desc), float(stock_len)

    used = []
    remaining = list(cuts_desc)

    # We treat kerf as consumed per piece cut off the bar (simple estimator-friendly model)
    rem = capacity
    i = 0
    while i < len(remaining):
        c = float(remaining[i])
        need = c + kerf
        if need <= rem + 1e-9:
            used.append(c)
            rem -= need
            remaining.pop(i)
            # restart best-fit search for next piece
            i = 0
        else:
            i += 1

    waste = max(0.0, rem)
    return used, remaining, waste


def optimize_structural_nesting_mix(
    cuts_with_qty,
    stock_options,
    kerf=0.0,
    end_trim=0.0,
    objective="waste",
    n_trials=600,
    seed=13,
):
    """Robust 1D nesting optimizer that can mix multiple stock lengths.

    Args:
        cuts_with_qty: list of dicts or tuples: [{length:float, qty:int}, ...] or [(len, qty), ...]
        stock_options: list of dicts: [{length:float, qty:int|None, cost:float|None}, ...]
            - qty None/0 => unlimited
        kerf: inches of material lost per cut (approx. per piece)
        end_trim: inches reserved per bar (clamp/trim allowance)
        objective: "waste" | "bars" | "cost" | "balanced"
        n_trials: randomized trials to improve solution quality
        seed: rng seed (deterministic)

    Returns:
        solution dict with keys:
          - bars: list of {stock_len, cuts, waste, used_len}
          - by_stock: list of {stock_len, bars, total_used, total_waste, utilization}
          - totals: {bars_used, total_waste, total_stock_len, utilization, total_cost}
    """
    import random

    # Normalize / validate inputs
    cuts = _expand_cuts_from_rows(cuts_with_qty)
    cuts = [float(c) for c in cuts if float(c) > 0]
    if not cuts:
        return {"bars": [], "by_stock": [], "totals": {"bars_used": 0, "total_waste": 0.0, "utilization": 0.0, "total_cost": 0.0}}

    opts = []
    for o in (stock_options or []):
        try:
            L = float(o.get("length", 0.0) or 0.0)
            if L <= 0:
                continue
            qty = o.get("qty", None)
            qty = None if qty in (None, "", 0) else int(qty)
            cost = o.get("cost", None)
            cost = None if cost in (None, "") else float(cost)
            opts.append({"length": L, "qty": qty, "cost": cost})
        except Exception:
            continue
    if not opts:
        return {"bars": [], "by_stock": [], "totals": {"bars_used": 0, "total_waste": 0.0, "utilization": 0.0, "total_cost": 0.0}}

    # Remove impossible cuts (longer than every usable stock minus trim)
    max_cap = max(float(o["length"]) - float(end_trim or 0.0) for o in opts)
    feasible = [c for c in cuts if c + float(kerf or 0.0) <= max_cap + 1e-9]
    infeasible = [c for c in cuts if c not in feasible]
    cuts = feasible
    if not cuts:
        return {
            "bars": [],
            "by_stock": [],
            "totals": {"bars_used": 0, "total_waste": 0.0, "utilization": 0.0, "total_cost": 0.0},
            "infeasible_cuts": infeasible,
        }

    rng = random.Random(int(seed))

    def score_solution(bars):
        total_waste = sum(b["waste"] for b in bars)
        bars_used = len(bars)
        total_cost = 0.0
        for b in bars:
            L = b["stock_len"]
            # cost per bar if provided, otherwise 0
            cost = 0.0
            for o in opts:
                if abs(o["length"] - L) < 1e-9 and o.get("cost") is not None:
                    cost = float(o["cost"])
                    break
            total_cost += cost

        if objective == "bars":
            return (bars_used, total_waste, total_cost)
        if objective == "cost":
            return (total_cost, total_waste, bars_used)
        if objective == "balanced":
            # estimator-friendly weighting: prefer fewer bars, then waste, then cost
            return (bars_used * 100000.0 + total_waste * 10.0 + total_cost)
        # default: minimize waste, then bars, then cost
        return (total_waste, bars_used, total_cost)

    def build_trial(cuts_list, opts_list):
        # Track remaining stock quantities
        qty_left = {}
        for o in opts_list:
            qty_left[o["length"]] = None if o.get("qty") is None else int(o.get("qty"))

        remaining = sorted(cuts_list, reverse=True)
        bars = []

        while remaining:
            biggest = remaining[0]

            # Choose best stock length for this step
            candidates = []
            for o in opts_list:
                L = float(o["length"])
                q = qty_left.get(L, None)
                if q is not None and q <= 0:
                    continue
                if biggest + float(kerf or 0.0) > (L - float(end_trim or 0.0)) + 1e-9:
                    continue

                used, rem_after, waste = _pack_one_bar_best_fit(remaining, L, kerf=kerf, end_trim=end_trim)
                if not used:
                    continue

                used_len = sum(used) + float(kerf or 0.0) * len(used)
                # Local heuristic score: waste first, then prefer using more material
                local = (waste, -used_len)
                candidates.append((local, L, used, rem_after, waste, used_len))

            if not candidates:
                # Shouldn't happen due to feasibility filter, but guard anyway
                break

            candidates.sort(key=lambda x: x[0])
            # Small randomness to escape local minima
            pick_idx = 0
            if len(candidates) > 1:
                pick_idx = 0 if rng.random() < 0.80 else min(len(candidates) - 1, 1)
            _, chosen_L, used, rem_after, waste, used_len = candidates[pick_idx]

            bars.append({
                "stock_len": float(chosen_L),
                "cuts": used,
                "used_len": float(used_len),
                "waste": float(waste),
            })

            remaining = sorted(rem_after, reverse=True)

            q = qty_left.get(chosen_L, None)
            if q is not None:
                qty_left[chosen_L] = q - 1

        return bars

    best_bars = None
    best_score = None

    # Base deterministic trial first
    base = build_trial(cuts, sorted(opts, key=lambda o: o["length"], reverse=True))
    best_bars = base
    best_score = score_solution(base)

    # Randomized improvements
    for _ in range(max(0, int(n_trials))):
        trial_cuts = list(cuts)
        rng.shuffle(trial_cuts)
        # Bias: keep larger pieces early
        trial_cuts = sorted(trial_cuts[:], reverse=True) if rng.random() < 0.6 else trial_cuts

        trial_opts = list(opts)
        if rng.random() < 0.7:
            # prefer larger stock first most of the time
            trial_opts.sort(key=lambda o: o["length"], reverse=True)
        else:
            rng.shuffle(trial_opts)

        bars = build_trial(trial_cuts, trial_opts)
        sc = score_solution(bars)
        if best_score is None or sc < best_score:
            best_score = sc
            best_bars = bars

    # Summaries
    bars = best_bars or []
    total_stock_len = sum(b["stock_len"] for b in bars)
    total_waste = sum(b["waste"] for b in bars)
    total_used = sum(b["used_len"] for b in bars)
    utilization = (total_used / total_stock_len) if total_stock_len > 0 else 0.0

    # cost
    total_cost = 0.0
    for b in bars:
        L = b["stock_len"]
        for o in opts:
            if abs(o["length"] - L) < 1e-9 and o.get("cost") is not None:
                total_cost += float(o["cost"])
                break

    # group by stock length
    by = {}
    for b in bars:
        L = b["stock_len"]
        by.setdefault(L, {"stock_len": L, "bars": 0, "total_used": 0.0, "total_waste": 0.0})
        by[L]["bars"] += 1
        by[L]["total_used"] += float(b["used_len"])
        by[L]["total_waste"] += float(b["waste"])
    by_stock = []
    for L in sorted(by.keys(), reverse=True):
        rec = by[L]
        tot = rec["bars"] * float(L)
        rec["utilization"] = (rec["total_used"] / tot) if tot > 0 else 0.0
        by_stock.append(rec)

    sol = {
        "bars": bars,
        "by_stock": by_stock,
        "totals": {
            "bars_used": len(bars),
            "total_waste": float(round(total_waste, 3)),
            "total_stock_len": float(round(total_stock_len, 3)),
            "total_used": float(round(total_used, 3)),
            "utilization": float(utilization),
            "total_cost": float(round(total_cost, 2)),
        },
    }
    if infeasible:
        sol["infeasible_cuts"] = infeasible
    return sol


def calculate_plate_nesting_yield(required_parts, stock_w, stock_h, allow_rot=True):
    if not required_parts or not stock_w > 0 or not stock_h > 0: return 0, []
    all_p = [];
    for part in required_parts:
        for _ in range(part['quantity']): all_p.append({'width':part['width'],'height':part['height'],'placed':False})
    all_p.sort(key=lambda p: p['height'], reverse=True)
    sheets = []
    while any(not p['placed'] for p in all_p):
        sheet_layout = {'width':stock_w, 'height':stock_h, 'parts':[]}
        shelves = []
        for part in all_p:
            if part['placed']: continue
            placed = False
            part_dims = [(part['width'],part['height'],False), (part['height'],part['width'],True)] if allow_rot else [(part['width'],part['height'],False)]
            for w, h, rot in part_dims:
                if w>stock_w or h>stock_h: continue
                on_shelf = False
                for shelf in shelves:
                    if h<=shelf['height'] and w<=shelf['width_left']:
                        x,y = stock_w - shelf['width_left'], shelf['y_pos']
                        sheet_layout['parts'].append({'x':x,'y':y,'width':w,'height':h,'rotated':rot})
                        shelf['width_left'] -= w
                        part['placed']=True; on_shelf=True; placed=True; break
                if on_shelf: break
                if not placed:
                    cy = sum(s['height'] for s in shelves)
                    if h <= (stock_h-cy):
                        ns = {'y_pos':cy,'height':h,'width_left':stock_w-w}; shelves.append(ns)
                        sheet_layout['parts'].append({'x':0,'y':ns['y_pos'],'width':w,'height':h,'rotated':rot})
                        part['placed']=True; placed=True; break
            if not placed: part['placed']=True; sheet_layout.setdefault('unplaced_parts',[]).append(part)
        sheets.append(sheet_layout)
    return len(sheets), sheets


# ============================================================
# Plate nesting optimization (dynamic)
# ============================================================

def _expand_plate_parts(required_parts):
    """Expand [{'width','height','quantity',...}] into individual rectangles."""
    all_p = []
    for part in (required_parts or []):
        try:
            q = int(part.get("quantity", 1) or 1)
        except Exception:
            q = 1
        for i in range(max(q, 1)):
            all_p.append(
                {
                    "part_name": part.get("part_name", "Part"),
                    "width": float(part.get("width", 0.0) or 0.0),
                    "height": float(part.get("height", 0.0) or 0.0),
                    "source": part.get("source", ""),
                }
            )
    return all_p


def _shelf_pack_rectangles(
    rectangles,
    stock_w,
    stock_h,
    *,
    edge_margin=0.0,
    part_gap=0.0,
    allow_rot=True,
    sort_key="area",
):
    """Simple shelf packing.

    Coordinates returned are in inches, with (0,0) at bottom-left of the sheet.
    """
    if not rectangles or not stock_w > 0 or not stock_h > 0:
        return {"width": stock_w, "height": stock_h, "parts": [], "unplaced": rectangles or []}

    usable_w = max(float(stock_w) - 2.0 * float(edge_margin), 0.0)
    usable_h = max(float(stock_h) - 2.0 * float(edge_margin), 0.0)
    if usable_w <= 0 or usable_h <= 0:
        return {"width": stock_w, "height": stock_h, "parts": [], "unplaced": rectangles}

    def area(r):
        return float(r["width"]) * float(r["height"])

    def max_side(r):
        return max(float(r["width"]), float(r["height"]))

    def height(r):
        return float(r["height"])

    def width(r):
        return float(r["width"])

    key_map = {
        "area": area,
        "max_side": max_side,
        "height": height,
        "width": width,
    }
    kf = key_map.get(str(sort_key or "area"), area)

    items = [dict(r) for r in rectangles]
    items.sort(key=kf, reverse=True)

    sheet_layout = {"width": float(stock_w), "height": float(stock_h), "parts": []}
    shelves = []  # each: {y, height, x_cursor}
    unplaced = []

    for r in items:
        w0 = float(r["width"]) ; h0 = float(r["height"])
        if w0 <= 0 or h0 <= 0:
            continue

        # Try 0/90 only
        candidates = [(w0, h0, 0)]
        if allow_rot and abs(w0 - h0) > 1e-9:
            candidates.append((h0, w0, 90))

        placed = False
        for w, h, rot in candidates:
            # Apply inter-part gap by expanding the footprint. We keep the rendered
            # rectangle size as the real part size, but ensure spacing by packing
            # with a larger footprint.
            fw = w + float(part_gap)
            fh = h + float(part_gap)
            if fw > usable_w + 1e-9 or fh > usable_h + 1e-9:
                continue

            # Try existing shelves
            for shelf in shelves:
                if fh <= shelf["height"] + 1e-9 and (shelf["x_cursor"] + fw) <= usable_w + 1e-9:
                    x = float(edge_margin) + shelf["x_cursor"]
                    y = float(edge_margin) + shelf["y"]
                    sheet_layout["parts"].append(
                        {
                            "x": float(x),
                            "y": float(y),
                            "width": float(w),
                            "height": float(h),
                            "rotated": bool(rot == 90),
                            "rotation": int(rot),
                            "part_name": r.get("part_name", "Part"),
                            "source": r.get("source", ""),
                        }
                    )
                    shelf["x_cursor"] += fw
                    placed = True
                    break
            if placed:
                break

            # New shelf
            used_h = sum(s["height"] for s in shelves)
            if (used_h + fh) <= usable_h + 1e-9:
                shelf = {"y": used_h, "height": fh, "x_cursor": fw}
                shelves.append(shelf)
                x = float(edge_margin) + 0.0
                y = float(edge_margin) + shelf["y"]
                sheet_layout["parts"].append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                        "rotated": bool(rot == 90),
                        "rotation": int(rot),
                        "part_name": r.get("part_name", "Part"),
                        "source": r.get("source", ""),
                    }
                )
                placed = True
                break

        if not placed:
            unplaced.append(r)

    sheet_layout["unplaced_parts"] = unplaced
    return sheet_layout


def optimize_plate_nesting(
    required_parts,
    stock_sizes,
    *,
    edge_margin=0.0,
    part_gap=0.0,
    allow_rot=True,
    objective="min_sheets",
):
    """Try multiple heuristics and return the best solution.

    Parameters
    ----------
    required_parts : list of dict
        Each dict needs: width, height, quantity. Optional: part_name, source.
    stock_sizes : list of dict
        Each dict: {'width': W, 'height': H, 'qty': optional}.
        If qty omitted, treated as unlimited.

    Returns
    -------
    dict
        {
          'best': {...},
          'candidates': [...]
        }
    """
    rects = _expand_plate_parts(required_parts)
    if not rects:
        return {"best": None, "candidates": []}

    clean_stock = []
    for s in (stock_sizes or []):
        try:
            w = float(s.get("width", 0.0) or 0.0)
            h = float(s.get("height", 0.0) or 0.0)
            if w <= 0 or h <= 0:
                continue
            qty = s.get("qty", None)
            qty_i = None
            if qty not in (None, ""):
                try:
                    qty_i = int(qty)
                except Exception:
                    qty_i = None
            clean_stock.append({"width": w, "height": h, "qty": qty_i})
        except Exception:
            continue

    if not clean_stock:
        return {"best": None, "candidates": []}

    sort_orders = ["area", "max_side", "height", "width"]

    def _score(sol):
        # Lower is better
        sheets_used = int(sol.get("sheets_used", 10**9))
        utilization = float(sol.get("utilization", 0.0))
        waste = float(sol.get("total_waste_area", 10**18))
        # Primary objective: fewer sheets. Secondary: better utilization.
        if str(objective) == "max_utilization":
            return (-utilization, sheets_used, waste)
        return (sheets_used, -utilization, waste)

    candidates = []

    for sort_key in sort_orders:
        # Try each stock as the *primary* stock (we still allow mixing if qty limits)
        for primary in clean_stock:
            remaining = [dict(r) for r in rects]
            sheets = []
            stock_pool = [dict(s) for s in clean_stock]

            def _take_sheet_size():
                # Prefer primary while available
                for s in stock_pool:
                    if abs(s["width"] - primary["width"]) < 1e-9 and abs(s["height"] - primary["height"]) < 1e-9:
                        if s["qty"] is None or s["qty"] > 0:
                            if s["qty"] is not None:
                                s["qty"] -= 1
                            return s
                # Otherwise take any available (largest area first)
                stock_pool.sort(key=lambda x: float(x["width"]) * float(x["height"]), reverse=True)
                for s in stock_pool:
                    if s["qty"] is None or s["qty"] > 0:
                        if s["qty"] is not None:
                            s["qty"] -= 1
                        return s
                return None

            # Greedy: keep making sheets until all placed or we run out of stock
            while remaining:
                sheet_size = _take_sheet_size()
                if sheet_size is None:
                    break
                layout = _shelf_pack_rectangles(
                    remaining,
                    sheet_size["width"],
                    sheet_size["height"],
                    edge_margin=edge_margin,
                    part_gap=part_gap,
                    allow_rot=allow_rot,
                    sort_key=sort_key,
                )
                placed = layout.get("parts", [])
                if not placed:
                    # Nothing fits on this sheet size.
                    break
                sheets.append(layout)

                # Build new remaining list from unplaced parts
                remaining = [dict(r) for r in layout.get("unplaced_parts", [])]

            # Compute metrics
            total_part_area = 0.0
            total_stock_area = 0.0
            for sh in sheets:
                total_stock_area += float(sh["width"]) * float(sh["height"])
                for p in sh.get("parts", []):
                    total_part_area += float(p["width"]) * float(p["height"])

            utilization = (total_part_area / total_stock_area) if total_stock_area > 0 else 0.0
            waste = max(total_stock_area - total_part_area, 0.0)

            sol = {
                "sort_key": sort_key,
                "primary_stock": {"width": primary["width"], "height": primary["height"]},
                "sheets_used": int(len(sheets)),
                "sheets": sheets,
                "unplaced": remaining,
                "total_part_area": float(round(total_part_area, 3)),
                "total_stock_area": float(round(total_stock_area, 3)),
                "total_waste_area": float(round(waste, 3)),
                "utilization": float(round(utilization, 4)),
            }
            candidates.append(sol)

    # Pick best
    best = None
    for sol in candidates:
        if best is None or _score(sol) < _score(best):
            best = sol
    return {"best": best, "candidates": candidates}
def calculate_weld_totals(weld_entries):
    total_wire_weight_lbs = 0.0
    total_weld_time_hours = 0.0
    for entry in weld_entries:
        weld_size_key, length_in = entry.get('size'), entry.get('length', 0)
        is_preheat, is_cjp = entry.get('preheat', False), entry.get('cjp', False)
        if not weld_size_key or length_in <= 0: continue
        weld_properties = WELD_DATA.get(weld_size_key)
        if not weld_properties: continue
        lbs_per_ft, in_per_hr = weld_properties.get('lbs_per_ft', 0), weld_properties.get('in_per_hr', 0)
        wire_weight = (lbs_per_ft / 12.0) * length_in
        total_wire_weight_lbs += wire_weight
        if in_per_hr > 0:
            base_weld_time_hours = length_in / in_per_hr
            if is_preheat: base_weld_time_hours *= PRE_HEAT_MULTIPLIER
            if is_cjp: base_weld_time_hours *= CJP_MULTIPLIER
            total_weld_time_hours += base_weld_time_hours
    return round(total_wire_weight_lbs, 2), round(total_weld_time_hours, 2)
