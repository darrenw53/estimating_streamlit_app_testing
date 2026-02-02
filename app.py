import io
import os
import hmac
import datetime
from typing import Dict, Any, List, Tuple
import math

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

import logic
from dxf_plate import parse_dxf_plate_single_part_geometry, render_part_thumbnail_data_uri

APP_TITLE = "Estimating Calculator"

# ============================================================
# Setup time rules
#  - 0.50 hr (30 min) setup per unique group
#  - plate: group by (Thickness, Material) per machine type
#  - roll: group by (Thickness, Material) where Rolling Enabled
#  - saw: group by Shape Label (structural)
# ============================================================
SETUP_HOURS_PER_GROUP = 0.50
SETUP_MIN_PER_GROUP = SETUP_HOURS_PER_GROUP * 60.0  # 30.0


# ============================================================
# Rolling model (Plate -> Optional)
# Estimated time driven primarily by part weight (calculated from plate parameters)
# NOTE: Rolling setup is NOT per row — it is computed as 0.50 hr per unique THK+MAT group in Summary
# ============================================================

ROLLING_OD_BUCKETS = ["<= 24 in", "24–60 in", "60–120 in", "> 120 in"]

ROLLING_WEIGHT_BUCKETS_HR = [
    (0.0, 100.0, 0.25),
    (100.0, 250.0, 0.40),
    (250.0, 500.0, 0.60),
    (500.0, 1000.0, 0.90),
    (1000.0, 2000.0, 1.40),
    (2000.0, 4000.0, 2.20),
    (4000.0, 8000.0, 3.20),
    (8000.0, 12000.0, 4.25),
]
ROLLING_OVER_12000_BASE_HR = 4.25
ROLLING_OVER_12000_PER_4000_HR = 0.60

ROLLING_OD_MULT = {
    "<= 24 in": 1.30,
    "24–60 in": 1.00,
    "60–120 in": 0.95,
    "> 120 in": 0.90,
}

def _rolling_thickness_multiplier(thk_in: float) -> float:
    t = float(thk_in or 0.0)
    if t < 0.25:
        return 0.85
    if t < 0.50:
        return 1.00
    if t < 1.00:
        return 1.25
    return 1.60

ROLLING_TYPE_MULT = {"Cylinder": 1.00, "Cone": 1.40}
ROLLING_PREBEND_MULT = 1.20
ROLLING_TOL_MULT = 1.10

def _rolling_base_hours_by_weight(weight_lbs: float) -> float:
    w = max(0.0, float(weight_lbs or 0.0))
    for lo, hi, hr in ROLLING_WEIGHT_BUCKETS_HR:
        if w <= hi and w > lo:
            return float(hr)
        if w == 0.0 and lo == 0.0:
            return float(hr)

    if w > 12000.0:
        extra = w - 12000.0
        steps = math.ceil(extra / 4000.0)
        return float(ROLLING_OVER_12000_BASE_HR + steps * ROLLING_OVER_12000_PER_4000_HR)

    for _, hi, hr in ROLLING_WEIGHT_BUCKETS_HR:
        if w <= hi:
            return float(hr)
    return float(ROLLING_OVER_12000_BASE_HR)

def calculate_rolling_time_minutes_per_item(
    weight_lbs: float,
    thickness_in: float,
    roll_type: str,
    od_bucket: str,
    prebend: bool,
    tight_tolerance: bool,
) -> Tuple[float, float, str]:
    roll_type = "Cone" if str(roll_type).lower().startswith("cone") else "Cylinder"
    base_hr = _rolling_base_hours_by_weight(weight_lbs)

    mult = 1.0
    mult *= _rolling_thickness_multiplier(thickness_in)
    mult *= float(ROLLING_OD_MULT.get(od_bucket, 1.0))
    mult *= float(ROLLING_TYPE_MULT.get(roll_type, 1.0))
    if prebend:
        mult *= float(ROLLING_PREBEND_MULT)
    if tight_tolerance:
        mult *= float(ROLLING_TOL_MULT)

    minutes_item = round(base_hr * mult * 60.0, 2)

    details = (
        f"base={base_hr:.2f}hr @ {float(weight_lbs):.0f}lb; "
        f"ODx={ROLLING_OD_MULT.get(od_bucket, 1.0):.2f}; "
        f"THKx={_rolling_thickness_multiplier(thickness_in):.2f}; "
        f"typex={ROLLING_TYPE_MULT.get(roll_type, 1.0):.2f}; "
        f"prebend={'Y' if prebend else 'N'}; tol={'Y' if tight_tolerance else 'N'}; "
        f"mult={mult:.2f}"
    )
    return minutes_item, mult, details


# ============================================================
# Session + Auth
# ============================================================

def _init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("estimate_parts", [])
    st.session_state.setdefault("plate_yield_results", {})
    st.session_state.setdefault("structural_yield_results", {})

def _get_password_from_secrets_or_env() -> str:
    if "auth" in st.secrets and "password" in st.secrets["auth"]:
        return str(st.secrets["auth"]["password"])
    return os.getenv("ESTIMATOR_APP_PASSWORD", "")

def require_auth() -> None:
    password = _get_password_from_secrets_or_env()
    if not password:
        st.session_state["authenticated"] = True
        return
    if st.session_state.get("authenticated"):
        return

    st.title(APP_TITLE)
    st.subheader("Sign in")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter")
    if ok:
        if hmac.compare_digest(entered, password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

@st.cache_data(show_spinner=False)
def _load_aisc_once(csv_path: str) -> bool:
    logic.aisc_data_load_attempted = False
    logic.AISC_TYPES_TO_LABELS_MAP = None
    logic.AISC_LABEL_TO_PROPERTIES_MAP = None
    logic.load_aisc_database(csv_path)
    return bool(logic.AISC_TYPES_TO_LABELS_MAP)


def _create_yield_image(sheet_layout: Dict[str, Any], scale: int = 5) -> Image.Image:
    padding = 20
    stock_w_px = int(sheet_layout["width"] * scale)
    stock_h_px = int(sheet_layout["height"] * scale)
    img_w = stock_w_px + 2 * padding
    img_h = stock_h_px + 2 * padding
    img = Image.new("RGB", (img_w, img_h), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle(
        [padding, padding, padding + stock_w_px, padding + stock_h_px],
        outline="black",
        width=2,
    )

    palette = ("#d1ecf1", "#f8d7da", "#e2e3e5", "#d4edda", "#fff3cd", "#d6d8db")
    for i, part in enumerate(sheet_layout.get("parts", [])):
        x0 = padding + int(part["x"] * scale)
        y0 = padding + int(part["y"] * scale)
        x1 = x0 + int(part["width"] * scale)
        y1 = y0 + int(part["height"] * scale)
        draw.rectangle([x0, y0, x1, y1], fill=palette[i % len(palette)], outline="black", width=1)
        if part["width"] * scale > 40 and part["height"] * scale > 15:
            draw.text((x0 + 5, y0 + 5), f"{part['width']:.1f}x{part['height']:.1f}", fill="black", font=font)
    return img


def _add_part(part: Dict[str, Any]) -> None:
    st.session_state["estimate_parts"].append(part)

def _clear_estimate() -> None:
    st.session_state["estimate_parts"] = []
    st.session_state["plate_yield_results"] = {}
    st.session_state["structural_yield_results"] = {}


def _export_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""

    preferred_order = [
        "Estimation Type",
        "Part Name",
        "Quantity",
        "Material",
        "Thickness (in)",
        "Width (in)",
        "Length (in)",
        "Structural Type",
        "Shape Label",
        "Mitered Cut",
        "Burn Machine Type",
        "Setup Category",
        "Setup Groups",
        "Setup Time (min)",
        "Drill Details Summary",
        "Weld Details Summary",
        "Bends (per item)",
        "Bend Complexity",
        "Perimeter (in/item)",
        "Feedrate (IPM)",
        "Weight per Foot (lbs/ft)",
        "Cross-sectional Area (in^2)",
        "Net Weight (lbs/item)",
        "Gross Weight (lbs/item)",
        "Burning Time (min/item)",
        "Drilling Time (min/item)",
        "Bend Time (min/item)",
        "Rolling Enabled",
        "Rolling Type",
        "Rolling OD Bucket",
        "Rolling Prebend",
        "Rolling Tight Tolerance",
        "Rolling Time (min/item)",
        "Rolling Details",
        "Total Rolling Run Time (min)",
        "Cutting Time (min/item)",
        "Fit Time (min/item)",
        "Total Gross Weight (lbs)",
        "Total Burning Run Time (min)",
        "Total Burning Setup Time (min)",
        "Total Drilling Time (min)",
        "Total Bend Time (min)",
        "Total Rolling Run Time (min)",
        "Total Rolling Setup Time (min)",
        "Total Cutting Run Time (min)",
        "Total Cutting Setup Time (min)",
        "Total Fit Time (min)",
        "Total Weld Wire (lbs)",
        "Total Weld Time (hours)",
    ]

    all_keys = set().union(*[r.keys() for r in rows])
    header = [k for k in preferred_order if k in all_keys] + sorted([k for k in all_keys if k not in preferred_order])

    import csv
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


# ============================================================
# Setup computation + setup rows
# ============================================================

def _compute_setup_times(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns setup minutes and group counts for:
      - Laser setup (plate): unique (thk, mat) among Laser plates
      - Kinetic setup (plate): unique (thk, mat) among Kinetic plates
      - Rolling setup (plate): unique (thk, mat) among rolling-enabled plates
      - Saw setup (structural): unique Shape Label among structural entries
    """
    laser_keys = set()
    kinetic_keys = set()
    rolling_keys = set()
    saw_keys = set()

    for r in rows:
        et = r.get("Estimation Type")

        if et == "Plate":
            thk = r.get("Thickness (in)", None)
            mat = r.get("Material", None)
            if thk is not None and mat is not None:
                key = (float(thk), str(mat))
                bm = str(r.get("Burn Machine Type", "") or "")
                if bm == "Laser":
                    laser_keys.add(key)
                elif bm == "Kinetic":
                    kinetic_keys.add(key)

                roll_enabled = str(r.get("Rolling Enabled", "No")).strip().lower() in ("yes", "y", "true", "1")
                if roll_enabled:
                    rolling_keys.add(key)

        elif et == "Structural":
            shape = r.get("Shape Label", None)
            if shape:
                saw_keys.add(str(shape))

    return {
        "laser_setup_groups": len(laser_keys),
        "kinetic_setup_groups": len(kinetic_keys),
        "rolling_setup_groups": len(rolling_keys),
        "saw_setup_groups": len(saw_keys),
        "laser_setup_min": len(laser_keys) * SETUP_MIN_PER_GROUP,
        "kinetic_setup_min": len(kinetic_keys) * SETUP_MIN_PER_GROUP,
        "rolling_setup_min": len(rolling_keys) * SETUP_MIN_PER_GROUP,
        "saw_setup_min": len(saw_keys) * SETUP_MIN_PER_GROUP,
    }


def _make_setup_rows(setup: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Creates 4 "setup line items" shown in the Summary table + included in CSV export.
    These rows should NOT be included in future setup computations (Estimation Type = 'Setup').
    """
    def row(name: str, groups: int, minutes: float) -> Dict[str, Any]:
        return {
            "Estimation Type": "Setup",
            "Part Name": name,
            "Quantity": 1,
            "Setup Category": name,
            "Setup Groups": int(groups),
            "Setup Time (min)": float(minutes),
            # Totals: keep explicit machine setup fields so you can roll up easily
            "Total Burning Setup Time (min)": float(minutes) if "Laser" in name or "Kinetic" in name else 0.0,
            "Total Rolling Setup Time (min)": float(minutes) if "Rolling" in name else 0.0,
            "Total Cutting Setup Time (min)": float(minutes) if "Saw" in name else 0.0,
            # Keep other totals columns present (0)
            "Total Burning Run Time (min)": 0.0,
            "Total Rolling Run Time (min)": 0.0,
            "Total Cutting Run Time (min)": 0.0,
            "Total Drilling Time (min)": 0.0,
            "Total Bend Time (min)": 0.0,
            "Total Fit Time (min)": 0.0,
            "Total Gross Weight (lbs)": 0.0,
            "Total Weld Wire (lbs)": 0.0,
            "Total Weld Time (hours)": 0.0,
        }

    return [
        row("Laser Setup", setup["laser_setup_groups"], setup["laser_setup_min"]),
        row("Kinetic Setup", setup["kinetic_setup_groups"], setup["kinetic_setup_min"]),
        row("Saw Setup", setup["saw_setup_groups"], setup["saw_setup_min"]),
        row("Rolling Setup", setup["rolling_setup_groups"], setup["rolling_setup_min"]),
    ]


# ============================================================
# Pages
# ============================================================

def page_plate() -> None:
    st.header("Plate")

    # ------------------------------------------------------------
    # DXF Batch Import (Plate Only)
    # ------------------------------------------------------------
    with st.expander("DXF Batch Import (Plate Only)", expanded=False):
        st.caption(
            "Upload one or more DXF files and we'll auto-detect plate size, true cut perimeter, and holes. "
            "Then enter thickness/material/qty and add to the estimate."
        )

        up_cols = st.columns([1, 1, 1, 2])
        with up_cols[0]:
            dxf_units = st.selectbox("DXF units", options=["Inches", "mm"], index=0)
        with up_cols[1]:
            flatten_tol = st.number_input("Curve tolerance (in)", min_value=0.001, value=0.01, step=0.001)
        with up_cols[2]:
            dxf_scale = st.number_input(
                "Scale multiplier",
                min_value=0.0001,
                value=1.0,
                step=0.1,
                help="Applied after unit conversion. Example: if DXF geometry is at 1/8 scale, enter 8.0.",
            )
        with up_cols[3]:
            ignore_layers_text = st.text_input(
                "Ignore layers containing (comma-separated)",
                value="ETCH,SCRIBE,ENGRAVE,MARK,TEXT,DIM,CENTER,CL",
            )

        strict_single = st.checkbox(
            "Strict single-part mode (fail if multiple outer profiles are found)",
            value=False,
            help="When enabled, the DXF import will error if the file appears to contain multiple parts / outer profiles.",
        )

        uploaded = st.file_uploader(
            "Upload DXF files",
            type=["dxf", "DXF"],
            accept_multiple_files=True,
        )

        if uploaded:
            ignore_sub = [s.strip() for s in ignore_layers_text.split(",") if s.strip()]
            units_key = "mm" if dxf_units.lower().startswith("mm") else "in"

            detected_rows = []
            for f in uploaded:
                try:
                    geom = parse_dxf_plate_single_part_geometry(
                        f.getvalue(),
                        filename=f.name,
                        units=units_key,
                        scale=float(dxf_scale),
                        ignore_layer_substrings=ignore_sub,
                        flatten_tol=float(flatten_tol),
                        strict_single=bool(strict_single),
                    )
                    if geom is None:
                        st.warning(f"{f.name}: no closed profile detected (after layer filtering).")
                        continue

                    part = geom.part
                    try:
                        preview_uri = render_part_thumbnail_data_uri(geom, size_px=140)
                    except Exception:
                        preview_uri = ""

                    detected_rows.append(
                        {
                            "Source DXF": f.name,
                            "Preview": preview_uri,
                            "Part Name": part.part_name,
                            "Width (in)": float(part.bbox_w_in),
                            "Length (in)": float(part.bbox_l_in),
                            "True Cut Perimeter (in)": float(part.cut_perimeter_in),
                            "Hole Count": int(part.hole_count),
                            "Total Hole Circumference (in)": float(part.hole_circumference_in),
                            "Thickness (in)": float(logic.THICKNESS_LIST[0]),
                            "Grade": str(logic.MATERIALS_LIST[0]),
                            "Quantity": 1,
                        }
                    )
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")

            if detected_rows:
                df = pd.DataFrame(detected_rows)
                st.markdown("#### Detected parts")
                edited = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="fixed",
                    column_config={
                        "Preview": st.column_config.ImageColumn("Preview", width="small"),
                        "Thickness (in)": st.column_config.SelectboxColumn(
                            "Thickness (in)",
                            options=[float(x) for x in logic.THICKNESS_LIST],
                            required=True,
                        ),
                        "Grade": st.column_config.SelectboxColumn(
                            "Grade",
                            options=[str(x) for x in logic.MATERIALS_LIST],
                            required=True,
                        ),
                        "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1, required=True),
                    },
                )

                add_all = st.button("Add all detected plate parts to estimate")
                if add_all:
                    added_n = 0
                    for _, r in edited.iterrows():
                        try:
                            thickness = float(r["Thickness (in)"])
                            material = str(r["Grade"])
                            quantity = int(r["Quantity"])

                            width = float(r["Width (in)"])
                            length = float(r["Length (in)"])
                            perimeter = float(r["True Cut Perimeter (in)"])

                            burn_machine = logic.get_plate_burn_machine_type(thickness)
                            feedrate = logic.get_feedrate_for_thickness(thickness, logic.FEEDRATE_TABLE_IPM)

                            drilling_time_item = 0.0
                            drill_summary_str = ""
                            burn_time_item = round(logic.calculate_burning_time(perimeter, feedrate), 2)
                            bend_time_item = 0.0

                            net_weight_item = logic.calculate_plate_net_weight(
                                thickness, width, length, logic.DENSITY_FACTOR_FOR_CALCULATION
                            )
                            gross_weight_item = logic.calculate_gross_weight(
                                net_weight_item, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT
                            )
                            fit_time_item = logic.calculate_fit_time(net_weight_item)

                            # Rolling defaults off for DXF import
                            total_roll_run = 0.0

                            part = {
                                "Estimation Type": "Plate",
                                "Part Name": str(r["Part Name"]),
                                "Quantity": quantity,
                                "Material": material,
                                "Thickness (in)": thickness,
                                "Width (in)": width,
                                "Length (in)": length,
                                "Bends (per item)": 0,
                                "Bend Complexity": "N/A",
                                "Burn Machine Type": burn_machine,
                                "Perimeter (in/item)": round(perimeter, 2),
                                "Feedrate (IPM)": float(feedrate),
                                "Drilling Time (min/item)": float(drilling_time_item),
                                "Drill Details Summary": drill_summary_str,
                                "Burning Time (min/item)": float(burn_time_item),
                                "Bend Time (min/item)": float(bend_time_item),
                                "Net Weight (lbs/item)": round(net_weight_item, 2),
                                "Gross Weight (lbs/item)": round(gross_weight_item, 2),
                                "Fit Time (min/item)": float(fit_time_item),

                                # Rolling fields
                                "Rolling Enabled": "No",
                                "Rolling Type": "N/A",
                                "Rolling OD Bucket": "N/A",
                                "Rolling Prebend": "No",
                                "Rolling Tight Tolerance": "No",
                                "Rolling Time (min/item)": 0.0,
                                "Rolling Details": "",
                                "Total Rolling Run Time (min)": float(total_roll_run),

                                # Totals (RUN ONLY — setup computed on Summary page)
                                "Total Gross Weight (lbs)": round(gross_weight_item * quantity, 2),
                                "Total Burning Run Time (min)": round(burn_time_item * quantity, 2),
                                "Total Drilling Time (min)": 0.0,
                                "Total Bend Time (min)": 0.0,
                                "Total Rolling Run Time (min)": 0.0,
                                "Total Fit Time (min)": round(fit_time_item * quantity, 2),

                                # DXF metrics
                                "DXF Source": str(r.get("Source DXF", "")),
                                "DXF Hole Count": int(r.get("Hole Count", 0)),
                                "DXF Total Hole Circumference (in)": float(r.get("Total Hole Circumference (in)", 0.0)),
                            }

                            _add_part(part)
                            added_n += 1
                        except Exception as e:
                            st.error(f"Could not add row '{r.get('Part Name','')}' to estimate: {e}")

                    if added_n:
                        st.success(f"Added {added_n} plate part(s) from DXF.")
                        st.rerun()
            else:
                st.warning("No closed profiles were detected in the uploaded DXFs.")

    # ------------------------------------------------------------
    # Manual Plate Entry (NOT a form)
    # ------------------------------------------------------------
    st.subheader("Manual plate entry")

    c1, c2 = st.columns(2)
    with c1:
        part_name = st.text_input("Part name", value="Unnamed Plate", key="plate_part_name")
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="plate_qty")
        material = st.selectbox("Material", options=logic.MATERIALS_LIST, index=0, key="plate_mat")
        thickness = st.selectbox("Thickness (in)", options=logic.THICKNESS_LIST, index=0, key="plate_thk")
    with c2:
        width = st.number_input("Width (in)", min_value=0.0, value=0.0, step=0.25, key="plate_w")
        length = st.number_input("Length (in)", min_value=0.0, value=0.0, step=0.25, key="plate_l")
        num_bends = st.number_input("Bends (per item)", min_value=0, value=0, step=1, key="plate_bends")
        bend_complexity = st.selectbox(
            "Bend complexity",
            options=["N/A"] + logic.BEND_COMPLEXITY_OPTIONS,
            index=0,
            key="plate_bend_cx",
        )
        if int(num_bends) == 0:
            bend_complexity = "N/A"

    st.markdown("#### Drilling (optional)")
    d1, d2, d3 = st.columns(3)
    with d1:
        hole_dia_1 = st.number_input("Hole 1 dia (in)", min_value=0.0, value=0.0, step=0.0625, key="h1d")
        hole_qty_1 = st.number_input("Hole 1 qty", min_value=0, value=0, step=1, key="h1q")
    with d2:
        hole_dia_2 = st.number_input("Hole 2 dia (in)", min_value=0.0, value=0.0, step=0.0625, key="h2d")
        hole_qty_2 = st.number_input("Hole 2 qty", min_value=0, value=0, step=1, key="h2q")
    with d3:
        hole_dia_3 = st.number_input("Hole 3 dia (in)", min_value=0.0, value=0.0, step=0.0625, key="h3d")
        hole_qty_3 = st.number_input("Hole 3 qty", min_value=0, value=0, step=1, key="h3q")

    st.markdown("#### Rolling (optional)")
    rolling_enabled = st.checkbox(
        "Rolling required",
        value=False,
        help="When enabled, rolling run time is calculated from plate weight + modifiers. Rolling setup is computed separately in Summary.",
        key="rolling_enabled",
    )

    roll_type = "Cylinder"
    roll_od_bucket = "24–60 in"
    roll_prebend = False
    roll_tight_tol = False

    if rolling_enabled:
        r1, r2, r3, r4 = st.columns([1.2, 1.4, 1.0, 1.2])
        with r1:
            roll_type = st.selectbox("Rolling type", options=["Cylinder", "Cone"], index=0, key="roll_type")
        with r2:
            roll_od_bucket = st.selectbox("OD bucket", options=ROLLING_OD_BUCKETS, index=1, key="roll_od")
        with r3:
            roll_prebend = st.checkbox("Prebend", value=False, key="roll_prebend")
        with r4:
            roll_tight_tol = st.checkbox("Tight tolerance", value=False, key="roll_tol")

        try:
            net_wt_preview = logic.calculate_plate_net_weight(
                float(thickness), float(width), float(length), logic.DENSITY_FACTOR_FOR_CALCULATION
            )
        except Exception:
            net_wt_preview = 0.0

        if net_wt_preview > 0:
            roll_time_item_prev, _, roll_details_prev = calculate_rolling_time_minutes_per_item(
                weight_lbs=net_wt_preview,
                thickness_in=float(thickness),
                roll_type=roll_type,
                od_bucket=roll_od_bucket,
                prebend=bool(roll_prebend),
                tight_tolerance=bool(roll_tight_tol),
            )
            st.caption(
                f"Rolling preview (net wt {net_wt_preview:.1f} lb): {roll_time_item_prev:.1f} min/item "
                f"(setup = {SETUP_MIN_PER_GROUP:.0f} min per unique THK+MAT group)"
            )

    add = st.button("Add plate to estimate", type="primary", key="plate_add_btn")
    if add:
        fake_form = {
            "hole_dia_1": hole_dia_1, "hole_qty_1": hole_qty_1,
            "hole_dia_2": hole_dia_2, "hole_qty_2": hole_qty_2,
            "hole_dia_3": hole_dia_3, "hole_qty_3": hole_qty_3,
        }

        burn_machine = logic.get_plate_burn_machine_type(thickness)
        perimeter = logic.calculate_plate_perimeter(width, length)
        feedrate = logic.get_feedrate_for_thickness(thickness, logic.FEEDRATE_TABLE_IPM)

        drilling_time_item, drill_summary_str = logic.process_plate_drilling_data(fake_form, thickness)
        burn_time_item = round(logic.calculate_burning_time(perimeter, feedrate), 2)
        bend_time_item = round(
            logic.calculate_bend_time(int(num_bends), bend_complexity, logic.BEND_TIME_PER_COMPLEXITY_MINUTES), 2
        )

        net_weight_item = logic.calculate_plate_net_weight(thickness, width, length, logic.DENSITY_FACTOR_FOR_CALCULATION)
        gross_weight_item = logic.calculate_gross_weight(net_weight_item, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT)
        fit_time_item = logic.calculate_fit_time(net_weight_item)

        if rolling_enabled:
            roll_time_item, _, roll_details = calculate_rolling_time_minutes_per_item(
                weight_lbs=net_weight_item,
                thickness_in=float(thickness),
                roll_type=roll_type,
                od_bucket=roll_od_bucket,
                prebend=bool(roll_prebend),
                tight_tolerance=bool(roll_tight_tol),
            )
            total_roll_run = round(roll_time_item * int(quantity), 2)
        else:
            roll_time_item = 0.0
            roll_details = ""
            total_roll_run = 0.0

        part = {
            "Estimation Type": "Plate",
            "Part Name": part_name,
            "Quantity": int(quantity),
            "Material": material,
            "Thickness (in)": float(thickness),
            "Width (in)": float(width),
            "Length (in)": float(length),
            "Bends (per item)": int(num_bends),
            "Bend Complexity": bend_complexity,
            "Burn Machine Type": burn_machine,
            "Perimeter (in/item)": round(perimeter, 2),
            "Feedrate (IPM)": float(feedrate),
            "Drilling Time (min/item)": float(drilling_time_item),
            "Drill Details Summary": drill_summary_str,
            "Burning Time (min/item)": float(burn_time_item),
            "Bend Time (min/item)": float(bend_time_item),
            "Net Weight (lbs/item)": round(net_weight_item, 2),
            "Gross Weight (lbs/item)": round(gross_weight_item, 2),
            "Fit Time (min/item)": float(fit_time_item),

            # Rolling fields
            "Rolling Enabled": "Yes" if rolling_enabled else "No",
            "Rolling Type": roll_type if rolling_enabled else "N/A",
            "Rolling OD Bucket": roll_od_bucket if rolling_enabled else "N/A",
            "Rolling Prebend": "Yes" if roll_prebend else "No",
            "Rolling Tight Tolerance": "Yes" if roll_tight_tol else "No",
            "Rolling Time (min/item)": float(roll_time_item),
            "Rolling Details": roll_details,
            "Total Rolling Run Time (min)": float(total_roll_run),

            # Totals (RUN ONLY — setup computed in Summary page)
            "Total Gross Weight (lbs)": round(gross_weight_item * quantity, 2),
            "Total Burning Run Time (min)": round(burn_time_item * quantity, 2),
            "Total Drilling Time (min)": round(drilling_time_item * quantity, 2),
            "Total Bend Time (min)": round(bend_time_item * quantity, 2),
            "Total Rolling Run Time (min)": round(total_roll_run, 2),
            "Total Fit Time (min)": round(fit_time_item * quantity, 2),
        }

        _add_part(part)
        st.success("Plate added.")
        st.rerun()


def page_structural() -> None:
    st.header("Structural")
    if not logic.AISC_TYPES_TO_LABELS_MAP:
        st.error("AISC shape database not loaded.")
        return

    structural_types = sorted(list(logic.AISC_TYPES_TO_LABELS_MAP.keys()))

    def _cleanup_old_shape_keys(current_type: str) -> None:
        prefix = "struct_shape_"
        for k in list(st.session_state.keys()):
            if k.startswith(prefix) and k != f"{prefix}{current_type}":
                del st.session_state[k]

    c1, c2 = st.columns(2)
    with c1:
        part_name = st.text_input("Part name", value="Unnamed Structural")
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        structural_type = st.selectbox("Type", options=structural_types, key="struct_type")
    with c2:
        labels = logic.AISC_TYPES_TO_LABELS_MAP.get(structural_type, [])
        _cleanup_old_shape_keys(structural_type)

        shape_key = f"struct_shape_{structural_type}"
        if labels and shape_key not in st.session_state:
            st.session_state[shape_key] = labels[0]
        shape_label = st.selectbox("Shape", options=labels, key=shape_key)
        length_in = st.number_input("Length (in)", min_value=0.0, value=0.0, step=1.0)
        is_mitered = st.checkbox("Mitered cut")

    add = st.button("Add structural to estimate")

    if add:
        props = logic.AISC_LABEL_TO_PROPERTIES_MAP.get(shape_label, {})
        weight_per_foot = float(props.get("W_float", 0.0))
        area_sq_in = float(props.get("A_float", 0.0))

        net_wt = logic.calculate_structural_piece_weight(weight_per_foot, length_in)
        gross_wt = logic.calculate_gross_weight(net_wt, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT)
        fit_t = logic.calculate_fit_time(net_wt)
        cut_t = logic.calculate_structural_cutting_time(
            area_sq_in, logic.STRUCTURAL_CUTTING_RATE_SQ_IN_PER_MIN, logic.STRUCTURAL_TRAVEL_TIME_PER_PIECE_MIN
        )
        if is_mitered:
            cut_t *= logic.MITER_CUT_MULTIPLIER

        part = {
            "Estimation Type": "Structural",
            "Part Name": part_name,
            "Quantity": int(quantity),
            "Structural Type": structural_type,
            "Shape Label": shape_label,
            "Length (in)": float(length_in),
            "Mitered Cut": "Yes" if is_mitered else "No",
            "Weight per Foot (lbs/ft)": weight_per_foot,
            "Cross-sectional Area (in^2)": area_sq_in,
            "Net Weight (lbs/item)": round(net_wt, 2),
            "Gross Weight (lbs/item)": round(gross_wt, 2),
            "Fit Time (min/item)": float(fit_t),
            "Cutting Time (min/item)": round(cut_t, 2),

            "Total Gross Weight (lbs)": round(gross_wt * quantity, 2),
            "Total Cutting Run Time (min)": round(cut_t * quantity, 2),
            "Total Fit Time (min)": round(fit_t * quantity, 2),

            # Keep these present
            "Burn Machine Type": "N/A",
            "Total Burning Run Time (min)": 0.0,
            "Total Drilling Time (min)": 0.0,
            "Total Bend Time (min)": 0.0,
            "Rolling Enabled": "No",
            "Total Rolling Run Time (min)": 0.0,
            "Drill Details Summary": "N/A",
        }
        _add_part(part)
        st.success("Structural added.")


def page_welding() -> None:
    st.header("Welding")
    st.caption("Enter welds (up to 10 shown). Add as a single summary line to the estimate.")

    with st.form("weld_form"):
        weld_entries = []
        total_length = 0.0
        preheat_count = 0
        cjp_count = 0

        for i in range(1, 11):
            cols = st.columns([2, 2, 1, 1])
            with cols[0]:
                size = st.selectbox(f"Weld size #{i}", options=[""] + logic.WELD_SIZE_OPTIONS, key=f"wsize_{i}")
            with cols[1]:
                length = st.number_input(f"Length (in) #{i}", min_value=0.0, value=0.0, step=1.0, key=f"wlen_{i}")
            with cols[2]:
                preheat = st.checkbox("Preheat", key=f"wpre_{i}")
            with cols[3]:
                cjp = st.checkbox("CJP", key=f"wcjp_{i}")

            if size and length > 0:
                weld_entries.append({"size": size, "length": float(length), "preheat": bool(preheat), "cjp": bool(cjp)})
                total_length += float(length)
                if preheat:
                    preheat_count += 1
                if cjp:
                    cjp_count += 1

        add = st.form_submit_button("Add welding summary")

    if add:
        total_wire_weight, total_time_hours = logic.calculate_weld_totals(weld_entries)
        weld_details_summary = (
            f"{len(weld_entries)} welds; TotLen: {total_length:.2f}in; "
            f"Pre-heats: {preheat_count}; CJPs: {cjp_count}"
        )
        part = {
            "Estimation Type": "Welding",
            "Quantity": 1,
            "Part Name": "Welding Task",
            "Weld Details Summary": weld_details_summary,
            "Total Weld Wire (lbs)": total_wire_weight,
            "Total Weld Time (hours)": total_time_hours,
            "Total Gross Weight (lbs)": 0.0,
            "Total Burning Run Time (min)": 0.0,
            "Total Drilling Time (min)": 0.0,
            "Total Bend Time (min)": 0.0,
            "Total Rolling Run Time (min)": 0.0,
            "Total Cutting Run Time (min)": 0.0,
            "Total Fit Time (min)": 0.0,
        }
        _add_part(part)
        st.success("Welding summary added.")


def _compute_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # IMPORTANT: setup rows (Estimation Type = 'Setup') are ignored here by design
    plate_wt = 0.0
    struct_wt = 0.0

    laser_run = 0.0
    kinetic_run = 0.0
    roll_run = 0.0
    saw_run = 0.0

    drill_t = 0.0
    bend_t = 0.0
    fit_t = 0.0

    weld_time_hr = 0.0
    weld_wire_lbs = 0.0

    for r in rows:
        et = r.get("Estimation Type")
        if et == "Plate":
            plate_wt += float(r.get("Total Gross Weight (lbs)", 0.0) or 0.0)
            drill_t += float(r.get("Total Drilling Time (min)", 0.0) or 0.0)
            bend_t += float(r.get("Total Bend Time (min)", 0.0) or 0.0)
            fit_t += float(r.get("Total Fit Time (min)", 0.0) or 0.0)
            roll_run += float(r.get("Total Rolling Run Time (min)", 0.0) or 0.0)

            bm = r.get("Burn Machine Type")
            if bm == "Laser":
                laser_run += float(r.get("Total Burning Run Time (min)", 0.0) or 0.0)
            elif bm == "Kinetic":
                kinetic_run += float(r.get("Total Burning Run Time (min)", 0.0) or 0.0)

        elif et == "Structural":
            struct_wt += float(r.get("Total Gross Weight (lbs)", 0.0) or 0.0)
            fit_t += float(r.get("Total Fit Time (min)", 0.0) or 0.0)
            saw_run += float(r.get("Total Cutting Run Time (min)", 0.0) or 0.0)

        elif et == "Welding":
            weld_time_hr += float(r.get("Total Weld Time (hours)", 0.0) or 0.0)
            weld_wire_lbs += float(r.get("Total Weld Wire (lbs)", 0.0) or 0.0)

        # 'Setup' rows ignored here

    setup = _compute_setup_times(rows)

    return {
        "plate_total_gross_weight": plate_wt,
        "structural_total_gross_weight": struct_wt,
        "combined_overall_gross_weight": plate_wt + struct_wt,

        # RUN times
        "laser_run_min": laser_run,
        "kinetic_run_min": kinetic_run,
        "saw_run_min": saw_run,
        "roll_run_min": roll_run,

        # SETUP times
        "laser_setup_min": setup["laser_setup_min"],
        "kinetic_setup_min": setup["kinetic_setup_min"],
        "saw_setup_min": setup["saw_setup_min"],
        "roll_setup_min": setup["rolling_setup_min"],

        # counts
        "laser_setup_groups": setup["laser_setup_groups"],
        "kinetic_setup_groups": setup["kinetic_setup_groups"],
        "saw_setup_groups": setup["saw_setup_groups"],
        "roll_setup_groups": setup["rolling_setup_groups"],

        # other
        "drill_min": drill_t,
        "bend_min": bend_t,
        "fit_min": fit_t,
        "weld_time_hr": weld_time_hr,
        "weld_wire_lbs": weld_wire_lbs,
    }


def page_summary() -> None:
    st.header("Summary")
    base_rows = st.session_state["estimate_parts"]
    if not base_rows:
        st.info("No parts yet. Add items in Plate / Structural / Welding.")
        return

    totals = _compute_totals(base_rows)
    setup = _compute_setup_times(base_rows)
    setup_rows = _make_setup_rows(setup)

    # Display rows = parts + setup line items
    display_rows = base_rows + setup_rows

    # Summary headline
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total gross weight (lbs)", f"{totals['combined_overall_gross_weight']:.2f}")
    c2.metric("Total fit time (min)", f"{totals['fit_min']:.2f}")
    c3.metric("Weld time (hr)", f"{totals['weld_time_hr']:.2f}")
    c4.metric("Weld wire (lbs)", f"{totals['weld_wire_lbs']:.2f}")

    st.subheader("Run vs Setup")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Laser RUN (min)", f"{totals['laser_run_min']:.2f}")
    r2.metric("Kinetic RUN (min)", f"{totals['kinetic_run_min']:.2f}")
    r3.metric("Saw RUN (min)", f"{totals['saw_run_min']:.2f}")
    r4.metric("Rolling RUN (min)", f"{totals['roll_run_min']:.2f}")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Laser SETUP (min)", f"{totals['laser_setup_min']:.2f}")
    s2.metric("Kinetic SETUP (min)", f"{totals['kinetic_setup_min']:.2f}")
    s3.metric("Saw SETUP (min)", f"{totals['saw_setup_min']:.2f}")
    s4.metric("Rolling SETUP (min)", f"{totals['roll_setup_min']:.2f}")

    with st.expander("Setup groups (sanity check)", expanded=False):
        st.write(
            {
                "Minutes per group": float(SETUP_MIN_PER_GROUP),
                "Laser groups": int(totals["laser_setup_groups"]),
                "Kinetic groups": int(totals["kinetic_setup_groups"]),
                "Saw groups": int(totals["saw_setup_groups"]),
                "Rolling groups": int(totals["roll_setup_groups"]),
            }
        )

    st.subheader("Estimate lines (including setup)")
    st.dataframe(display_rows, use_container_width=True)

    csv_bytes = _export_csv_bytes(display_rows)
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"estimate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_state()
    require_auth()

    ok = _load_aisc_once(logic.AISC_CSV_FILENAME)
    if not ok:
        st.warning("AISC database could not be loaded. Structural tab will not work until the CSV is present.")

    with st.sidebar:
        st.title(APP_TITLE)
        st.caption("Password-protected Streamlit app")
        page = st.radio("Go to", ["Plate", "Structural", "Welding", "Summary"], index=0)
        st.write("—")
        st.write(f"Items in estimate: **{len(st.session_state['estimate_parts'])}**")
        if st.button("Clear estimate", type="secondary"):
            _clear_estimate()
            st.rerun()
        if st.button("Sign out", type="secondary"):
            st.session_state["authenticated"] = False
            st.rerun()

    if page == "Plate":
        page_plate()
    elif page == "Structural":
        page_structural()
    elif page == "Welding":
        page_welding()
    else:
        page_summary()


if __name__ == "__main__":
    main()
