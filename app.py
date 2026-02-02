import io
import os
import hmac
import datetime
from typing import Dict, Any, List, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

import logic
from dxf_plate import parse_dxf_plate_single_part_geometry, render_part_thumbnail_data_uri


# ============================================================
# Streamlit App: Estimating Calculator (Plate / Structural / Welding)
# - Uses the existing calculation logic in logic.py
# - Stores an in-progress estimate in st.session_state
# - Optional password protection via Streamlit secrets
# ============================================================


APP_TITLE = "Estimating Calculator"


# ============================================================
# Rolling model (Plate -> Optional)
# - Estimated time driven primarily by part weight
# - Adds modifiers for OD bucket, thickness, prebend, and tight tolerance
# NOTE: These are starter defaults — you can tune once you see real shop feedback.
# ============================================================

ROLLING_OD_BUCKETS = ["<= 24 in", "24–60 in", "60–120 in", "> 120 in"]

# Base "run time per item" by weight bucket (minutes/item)
# Buckets: (max_weight_lbs, minutes)
ROLLING_BASE_CYLINDER: List[Tuple[float, float]] = [
    (250, 20),
    (750, 35),
    (1500, 55),
    (3000, 85),
    (6000, 130),
    (float("inf"), 180),
]

ROLLING_BASE_CONE: List[Tuple[float, float]] = [
    (250, 30),
    (750, 55),
    (1500, 85),
    (3000, 125),
    (6000, 185),
    (float("inf"), 250),
]

# Setup time per lot (minutes) — applied once per line item when rolling is enabled
ROLLING_SETUP_MIN = {
    "Cylinder": 30.0,
    "Cone": 45.0,
}

# OD multipliers
ROLLING_OD_MULT = {
    "<= 24 in": 1.00,
    "24–60 in": 1.10,
    "60–120 in": 1.20,
    "> 120 in": 1.35,
}

# Thickness multipliers (inches) — stacked as a single multiplier
def _rolling_thickness_multiplier(thk_in: float) -> float:
    if thk_in >= 2.0:
        return 1.35
    if thk_in >= 1.0:
        return 1.15
    return 1.00


def _rolling_base_minutes_per_item(weight_lbs: float, roll_type: str) -> float:
    table = ROLLING_BASE_CONE if roll_type == "Cone" else ROLLING_BASE_CYLINDER
    for max_wt, mins in table:
        if weight_lbs <= max_wt:
            return float(mins)
    return float(table[-1][1])


def calculate_rolling_time(
    weight_lbs: float,
    thickness_in: float,
    roll_type: str,
    od_bucket: str,
    prebend: bool,
    tight_tolerance: bool,
) -> Tuple[float, float, float, str]:
    """
    Returns:
        rolling_time_item_min,
        setup_min_lot,
        total_multiplier,
        details_string
    """
    weight_lbs = max(float(weight_lbs), 0.0)
    thickness_in = max(float(thickness_in), 0.0)
    roll_type = "Cone" if str(roll_type).lower().startswith("cone") else "Cylinder"

    base = _rolling_base_minutes_per_item(weight_lbs, roll_type)
    od_mult = float(ROLLING_OD_MULT.get(od_bucket, 1.0))
    thk_mult = float(_rolling_thickness_multiplier(thickness_in))
    prebend_mult = 1.15 if prebend else 1.00
    tol_mult = 1.25 if tight_tolerance else 1.00

    total_mult = od_mult * thk_mult * prebend_mult * tol_mult
    time_item = round(base * total_mult, 2)

    setup = float(ROLLING_SETUP_MIN.get(roll_type, 30.0))

    details = (
        f"base={base:.0f}min @ {weight_lbs:.0f}lb; "
        f"OD={od_mult:.2f}x; THK={thk_mult:.2f}x; "
        f"prebend={prebend_mult:.2f}x; tol={tol_mult:.2f}x"
    )
    return time_item, setup, total_mult, details


def _init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("estimate_parts", [])
    st.session_state.setdefault("plate_yield_results", {})
    st.session_state.setdefault("structural_yield_results", {})


def _get_password_from_secrets_or_env() -> str:
    """Return app password from Streamlit secrets (preferred) or env var as fallback."""
    # Preferred: Streamlit secrets
    #   [auth]
    #   password = "..."
    if "auth" in st.secrets and "password" in st.secrets["auth"]:
        return str(st.secrets["auth"]["password"])
    # Fallback for local dev only
    return os.getenv("ESTIMATOR_APP_PASSWORD", "")


def require_auth() -> None:
    """Gate the app behind a password (if configured)."""
    password = _get_password_from_secrets_or_env()

    # If no password is configured, run unlocked.
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
    """Load AISC database once and cache it."""
    logic.aisc_data_load_attempted = False
    logic.AISC_TYPES_TO_LABELS_MAP = None
    logic.AISC_LABEL_TO_PROPERTIES_MAP = None
    logic.load_aisc_database(csv_path)
    return bool(logic.AISC_TYPES_TO_LABELS_MAP)


def _create_yield_image(sheet_layout: Dict[str, Any], scale: int = 5) -> Image.Image:
    """Create a PIL image for a plate nesting layout (displayed in Streamlit)."""
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

    # Build stable header order (match your Flask export ordering as much as possible)
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
        "Rolling Setup (min/lot)",
        "Rolling Details",
        "Total Rolling Time (min)",
        "Cutting Time (min/item)",
        "Fit Time (min/item)",
        "Total Gross Weight (lbs)",
        "Total Burning Time (min)",
        "Total Drilling Time (min)",
        "Total Bend Time (min)",
        "Total Rolling Time (min)",
        "Total Cutting Time (min)",
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
                        "Preview": st.column_config.ImageColumn(
                            "Preview",
                            help="Auto-rendered thumbnail of the detected profile.",
                            width="small",
                        ),
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
                        "Quantity": st.column_config.NumberColumn(
                            "Quantity",
                            min_value=1,
                            step=1,
                            required=True,
                        ),
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

                            # No bends from DXF import (plate-only); user can edit later if desired.
                            bend_time_item = 0.0

                            net_weight_item = logic.calculate_plate_net_weight(
                                thickness, width, length, logic.DENSITY_FACTOR_FOR_CALCULATION
                            )
                            gross_weight_item = logic.calculate_gross_weight(
                                net_weight_item, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT
                            )
                            fit_time_item = logic.calculate_fit_time(net_weight_item)

                            # Rolling defaults for DXF import (off)
                            rolling_enabled = False
                            roll_type = "Cylinder"
                            roll_od_bucket = "24–60 in"
                            roll_prebend = False
                            roll_tight_tol = False
                            roll_time_item = 0.0
                            roll_setup_lot = 0.0
                            roll_details = ""
                            total_roll_time = 0.0

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
                                "Rolling Enabled": "Yes" if rolling_enabled else "No",
                                "Rolling Type": roll_type,
                                "Rolling OD Bucket": roll_od_bucket,
                                "Rolling Prebend": "Yes" if roll_prebend else "No",
                                "Rolling Tight Tolerance": "Yes" if roll_tight_tol else "No",
                                "Rolling Time (min/item)": float(roll_time_item),
                                "Rolling Setup (min/lot)": float(roll_setup_lot),
                                "Rolling Details": roll_details,
                                "Total Rolling Time (min)": float(total_roll_time),
                                "Total Gross Weight (lbs)": round(gross_weight_item * quantity, 2),
                                "Total Burning Time (min)": round(burn_time_item * quantity, 2),
                                "Total Drilling Time (min)": round(drilling_time_item * quantity, 2),
                                "Total Bend Time (min)": round(bend_time_item * quantity, 2),
                                "Total Rolling Time (min)": round(total_roll_time, 2),
                                "Total Fit Time (min)": round(fit_time_item * quantity, 2),
                                # DXF metrics (kept for transparency)
                                "DXF Source": str(r.get("Source DXF", "")),
                                "DXF Hole Count": int(r.get("Hole Count", 0)),
                                "DXF Total Hole Circumference (in)": float(
                                    r.get("Total Hole Circumference (in)", 0.0)
                                ),
                            }

                            _add_part(part)
                            added_n += 1
                        except Exception as e:
                            st.error(f"Could not add row '{r.get('Part Name','')}' to estimate: {e}")

                    if added_n:
                        st.success(f"Added {added_n} plate part(s) from DXF.")
                        st.rerun()
            else:
                st.warning("No closed profiles were detected in the uploaded DXFs (or everything was filtered out by ignored layers).")

    with st.form("plate_form"):
        c1, c2 = st.columns(2)
        with c1:
            part_name = st.text_input("Part name", value="Unnamed Plate")
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            material = st.selectbox("Material", options=logic.MATERIALS_LIST, index=0)
            thickness = st.selectbox("Thickness (in)", options=logic.THICKNESS_LIST, index=0)
        with c2:
            width = st.number_input("Width (in)", min_value=0.0, value=0.0, step=0.25)
            length = st.number_input("Length (in)", min_value=0.0, value=0.0, step=0.25)
            num_bends = st.number_input("Bends (per item)", min_value=0, value=0, step=1)
            bend_complexity = st.selectbox("Bend complexity", options=["N/A"] + logic.BEND_COMPLEXITY_OPTIONS, index=0)
            if int(num_bends) == 0:
                bend_complexity = "N/A"

        st.markdown("#### Drilling (optional)")
        d1, d2, d3 = st.columns(3)
        with d1:
            hole_dia_1 = st.number_input("Hole 1 dia (in)", min_value=0.0, value=0.0, step=0.0625)
            hole_qty_1 = st.number_input("Hole 1 qty", min_value=0, value=0, step=1)
        with d2:
            hole_dia_2 = st.number_input("Hole 2 dia (in)", min_value=0.0, value=0.0, step=0.0625)
            hole_qty_2 = st.number_input("Hole 2 qty", min_value=0, value=0, step=1)
        with d3:
            hole_dia_3 = st.number_input("Hole 3 dia (in)", min_value=0.0, value=0.0, step=0.0625)
            hole_qty_3 = st.number_input("Hole 3 qty", min_value=0, value=0, step=1)

        st.markdown("#### Rolling (optional)")
        rolling_enabled = st.checkbox(
            "Rolling required",
            value=False,
            help="Adds rolling time based on calculated plate weight + modifiers.",
        )
        roll_type = "Cylinder"
        roll_od_bucket = "24–60 in"
        roll_prebend = False
        roll_tight_tol = False

        if rolling_enabled:
            r1, r2, r3, r4 = st.columns([1.2, 1.2, 1.0, 1.0])
            with r1:
                roll_type = st.selectbox("Rolling type", options=["Cylinder", "Cone"], index=0)
            with r2:
                roll_od_bucket = st.selectbox("OD bucket", options=ROLLING_OD_BUCKETS, index=1)
            with r3:
                roll_prebend = st.checkbox("Prebend", value=False)
            with r4:
                roll_tight_tol = st.checkbox("Tight tolerance", value=False)

        add = st.form_submit_button("Add plate to estimate")

    if add:
        # Build a form-like dict for existing logic
        fake_form = {
            "hole_dia_1": hole_dia_1,
            "hole_qty_1": hole_qty_1,
            "hole_dia_2": hole_dia_2,
            "hole_qty_2": hole_qty_2,
            "hole_dia_3": hole_dia_3,
            "hole_qty_3": hole_qty_3,
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

        # Rolling calc (optional)
        if rolling_enabled:
            roll_time_item, roll_setup_lot, roll_mult, roll_details = calculate_rolling_time(
                weight_lbs=net_weight_item,  # use net weight as driver
                thickness_in=float(thickness),
                roll_type=roll_type,
                od_bucket=roll_od_bucket,
                prebend=bool(roll_prebend),
                tight_tolerance=bool(roll_tight_tol),
            )
            total_roll_time = round((roll_time_item * int(quantity)) + float(roll_setup_lot), 2)
        else:
            roll_time_item = 0.0
            roll_setup_lot = 0.0
            roll_details = ""
            total_roll_time = 0.0

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
            "Rolling Setup (min/lot)": float(roll_setup_lot),
            "Rolling Details": roll_details,
            "Total Rolling Time (min)": float(total_roll_time),
            # Totals
            "Total Gross Weight (lbs)": round(gross_weight_item * quantity, 2),
            "Total Burning Time (min)": round(burn_time_item * quantity, 2),
            "Total Drilling Time (min)": round(drilling_time_item * quantity, 2),
            "Total Bend Time (min)": round(bend_time_item * quantity, 2),
            "Total Rolling Time (min)": round(total_roll_time, 2),
            "Total Fit Time (min)": round(fit_time_item * quantity, 2),
        }
        _add_part(part)
        st.success("Plate added.")


def page_structural() -> None:
    st.header("Structural")
    if not logic.AISC_TYPES_TO_LABELS_MAP:
        st.error("AISC shape database not loaded.")
        return

    structural_types = sorted(list(logic.AISC_TYPES_TO_LABELS_MAP.keys()))

    # NOTE: We intentionally do NOT wrap these widgets in a st.form.
    # Streamlit forms do not rerun the script on widget changes, which breaks
    # dependent dropdowns (Type -> Shape). Keeping them outside the form ensures
    # the Shape list updates immediately when Type changes.

    # Streamlit can sometimes keep the previous selectbox value even when
    # the options list changes (especially with very large lists).
    # To make the dependent dropdown rock-solid, we give the Shape widget a
    # *type-specific key* so it remounts whenever Type changes.

    def _cleanup_old_shape_keys(current_type: str) -> None:
        prefix = "struct_shape_"
        for k in list(st.session_state.keys()):
            if k.startswith(prefix) and k != f"{prefix}{current_type}":
                # keep state tidy; not strictly required
                del st.session_state[k]

    c1, c2 = st.columns(2)
    with c1:
        part_name = st.text_input("Part name", value="Unnamed Structural")
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        structural_type = st.selectbox(
            "Type",
            options=structural_types,
            key="struct_type",
        )
    with c2:
        labels = logic.AISC_TYPES_TO_LABELS_MAP.get(structural_type, [])
        _cleanup_old_shape_keys(structural_type)

        shape_key = f"struct_shape_{structural_type}"
        # Ensure there is a valid default for this Type.
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
            "Total Cutting Time (min)": round(cut_t * quantity, 2),
            "Total Fit Time (min)": round(fit_t * quantity, 2),
            # Keep these so totals logic is simpler
            "Burn Machine Type": "N/A",
            "Burning Time (min/item)": 0.0,
            "Total Burning Time (min)": 0.0,
            "Bend Time (min/item)": 0.0,
            "Total Bend Time (min)": 0.0,
            "Drilling Time (min/item)": 0.0,
            "Total Drilling Time (min)": 0.0,
            "Drill Details Summary": "N/A",
            # Rolling fields (not applicable)
            "Rolling Enabled": "No",
            "Rolling Type": "N/A",
            "Rolling OD Bucket": "N/A",
            "Rolling Prebend": "No",
            "Rolling Tight Tolerance": "No",
            "Rolling Time (min/item)": 0.0,
            "Rolling Setup (min/lot)": 0.0,
            "Rolling Details": "",
            "Total Rolling Time (min)": 0.0,
        }
        _add_part(part)
        st.success("Structural added.")


def page_welding() -> None:
    st.header("Welding")
    st.caption("Enter welds (up to 50). Add as a single summary line to the estimate.")

    with st.form("weld_form"):
        weld_entries = []
        total_length = 0.0
        preheat_count = 0
        cjp_count = 0

        for i in range(1, 11):
            # (Most users only need a handful; increase to 50 if you want, but 10 is cleaner UI.)
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
            # Keep totals keys consistent
            "Total Gross Weight (lbs)": 0.0,
            "Total Burning Time (min)": 0.0,
            "Total Drilling Time (min)": 0.0,
            "Total Bend Time (min)": 0.0,
            "Total Rolling Time (min)": 0.0,
            "Total Cutting Time (min)": 0.0,
            "Total Fit Time (min)": 0.0,
        }
        _add_part(part)
        st.success("Welding summary added.")


def _compute_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    plate_wt = 0.0
    struct_wt = 0.0
    plt_bend_t = 0.0
    plt_roll_t = 0.0
    # Structural cutting time is treated as "saw" time in the UI totals.
    str_cut_t = 0.0
    fit_t = 0.0
    laser_burn_t = 0.0
    kinetic_burn_t = 0.0
    drill_t = 0.0
    weld_time_hr = 0.0
    weld_wire_lbs = 0.0
    perimeter_total_in = 0.0

    has_plate = any(r.get("Estimation Type") == "Plate" for r in rows)
    has_struct = any(r.get("Estimation Type") == "Structural" for r in rows)

    for r in rows:
        fit_t += float(r.get("Total Fit Time (min)", 0.0) or 0.0)

        # Perimeter is stored as inches per item; multiply by quantity where available.
        try:
            per_item = float(r.get("Perimeter (in/item)", 0.0) or 0.0)
            qty = int(r.get("Quantity", 0) or 0)
            perimeter_total_in += per_item * qty
        except Exception:
            pass

        etype = r.get("Estimation Type")
        if etype == "Plate":
            plate_wt += float(r.get("Total Gross Weight (lbs)", 0.0) or 0.0)
            plt_bend_t += float(r.get("Total Bend Time (min)", 0.0) or 0.0)
            plt_roll_t += float(r.get("Total Rolling Time (min)", 0.0) or 0.0)
            drill_t += float(r.get("Total Drilling Time (min)", 0.0) or 0.0)
            if r.get("Burn Machine Type") == "Laser":
                laser_burn_t += float(r.get("Total Burning Time (min)", 0.0) or 0.0)
            elif r.get("Burn Machine Type") == "Kinetic":
                kinetic_burn_t += float(r.get("Total Burning Time (min)", 0.0) or 0.0)
        elif etype == "Structural":
            struct_wt += float(r.get("Total Gross Weight (lbs)", 0.0) or 0.0)
            str_cut_t += float(r.get("Total Cutting Time (min)", 0.0) or 0.0)
        elif etype == "Welding":
            weld_time_hr += float(r.get("Total Weld Time (hours)", 0.0) or 0.0)
            weld_wire_lbs += float(r.get("Total Weld Wire (lbs)", 0.0) or 0.0)

    return {
        "plate_total_gross_weight": plate_wt,
        "structural_total_gross_weight": struct_wt,
        "grand_total_laser_burn_time": laser_burn_t,
        "grand_total_kinetic_burn_time": kinetic_burn_t,
        "grand_total_plate_drilling_time": drill_t,
        "grand_total_plate_bend_time": plt_bend_t,
        "grand_total_plate_rolling_time": plt_roll_t,
        # Back-compat key (older UI label)
        "grand_total_structural_cutting_time": str_cut_t,
        # Preferred key (explicit)
        "grand_total_saw_time": str_cut_t,
        "grand_total_fit_time": fit_t,
        "grand_total_weld_time_hours": weld_time_hr,
        "grand_total_weld_wire_lbs": weld_wire_lbs,
        "grand_total_perimeter_in": perimeter_total_in,
        "combined_overall_gross_weight": plate_wt + struct_wt,
        "has_plate_entries": has_plate,
        "has_structural_entries": has_struct,
    }


def page_summary() -> None:
    st.header("Summary")
    rows = st.session_state["estimate_parts"]
    if not rows:
        st.info("No parts yet. Add items in Plate / Structural / Welding.")
        return

    totals = _compute_totals(rows)

    # Primary headline totals
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total gross weight (lbs)", f"{totals['combined_overall_gross_weight']:.2f}")
    c2.metric("Total fit time (min)", f"{totals['grand_total_fit_time']:.2f}")
    c3.metric("Weld time (hr)", f"{totals['grand_total_weld_time_hours']:.2f}")
    c4.metric("Total perimeter (in)", f"{totals['grand_total_perimeter_in']:.2f}")

    # Time breakdown (requested): Saw / Laser / Kinetic (+ Rolling)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Saw time (min)", f"{totals.get('grand_total_saw_time', totals.get('grand_total_structural_cutting_time', 0.0)):.2f}")
    t2.metric("Laser time (min)", f"{totals['grand_total_laser_burn_time']:.2f}")
    t3.metric("Kinetic time (min)", f"{totals['grand_total_kinetic_burn_time']:.2f}")
    t4.metric("Rolling time (min)", f"{totals['grand_total_plate_rolling_time']:.2f}")

    with st.expander("More totals", expanded=False):
        st.write(
            {
                "Laser burn (min)": round(totals["grand_total_laser_burn_time"], 2),
                "Kinetic burn (min)": round(totals["grand_total_kinetic_burn_time"], 2),
                "Plate drilling (min)": round(totals["grand_total_plate_drilling_time"], 2),
                "Plate bend (min)": round(totals["grand_total_plate_bend_time"], 2),
                "Plate rolling (min)": round(totals["grand_total_plate_rolling_time"], 2),
                "Saw time (min)": round(totals.get("grand_total_saw_time", totals.get("grand_total_structural_cutting_time", 0.0)), 2),
                "Weld wire (lbs)": round(totals["grand_total_weld_wire_lbs"], 2),
                "Total perimeter (in)": round(totals["grand_total_perimeter_in"], 2),
            }
        )

    st.dataframe(rows, use_container_width=True)

    csv_bytes = _export_csv_bytes(rows)
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"estimate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.divider()

    # ---- Plate yield ----
    if totals["has_plate_entries"]:
        st.subheader("Plate yield")
        # Group plates by thickness + material
        grouped = {}
        for r in rows:
            if r.get("Estimation Type") != "Plate":
                continue
            key = f"{float(r.get('Thickness (in)', 0.0)):.4f}in_{r.get('Material', 'N/A')}"
            grouped.setdefault(key, {"material": r.get("Material"), "thickness": r.get("Thickness (in)"), "parts_list": []})
            grouped[key]["parts_list"].append(
                {
                    "width": float(r.get("Width (in)", 0.0) or 0.0),
                    "height": float(r.get("Length (in)", 0.0) or 0.0),
                    "quantity": int(r.get("Quantity", 0) or 0),
                }
            )

        with st.form("plate_yield"):
            stock_inputs = {}
            for key, g in grouped.items():
                st.markdown(f"**{g['material']} @ {float(g['thickness']):.4f} in**")
                c1, c2 = st.columns(2)
                with c1:
                    sw = st.number_input(f"Stock width (in) — {key}", min_value=0.0, value=60.0, step=1.0)
                with c2:
                    sl = st.number_input(f"Stock length (in) — {key}", min_value=0.0, value=120.0, step=1.0)
                stock_inputs[key] = (float(sw), float(sl))
                st.write("—")
            run = st.form_submit_button("Calculate plate yield")

        if run:
            results = {}
            for key, g in grouped.items():
                stock_w, stock_h = stock_inputs[key]
                sheets_needed, layouts = logic.calculate_plate_nesting_yield(g["parts_list"], stock_w, stock_h)
                total_parts_area = sum(p["width"] * p["height"] * p["quantity"] for p in g["parts_list"])
                total_stock_area = sheets_needed * stock_w * stock_h
                yield_pct = (total_parts_area / total_stock_area) * 100 if total_stock_area > 0 else 0.0
                results[key] = {
                    "stock_size": f"{stock_w:.2f}x{stock_h:.2f} in",
                    "sheets_needed": sheets_needed,
                    "yield_percent": yield_pct,
                    "layouts": layouts,
                    "unplaced_count": len(layouts[-1].get("unplaced_parts", [])) if layouts else 0,
                }
            st.session_state["plate_yield_results"] = results

        # Display saved results
        if st.session_state["plate_yield_results"]:
            for key, res in st.session_state["plate_yield_results"].items():
                st.markdown(f"**{key}**")
                st.write(
                    {
                        "Stock": res["stock_size"],
                        "Sheets needed": res["sheets_needed"],
                        "Yield %": round(res["yield_percent"], 2),
                        "Unplaced": res["unplaced_count"],
                    }
                )
                for i, layout in enumerate(res.get("layouts", []), start=1):
                    st.image(_create_yield_image(layout), caption=f"Sheet {i}")

    # ---- Structural yield ----
    if totals["has_structural_entries"]:
        st.subheader("Structural yield")
        # Group by shape label
        grouped_struct = {}
        for r in rows:
            if r.get("Estimation Type") != "Structural":
                continue
            label = r.get("Shape Label")
            if not label:
                continue
            grouped_struct.setdefault(label, []).append(
                {"length": float(r.get("Length (in)", 0.0)), "quantity": int(r.get("Quantity", 0))}
            )

        with st.form("struct_yield"):
            stock_len_inputs = {}
            for label in grouped_struct:
                stock_len_inputs[label] = st.text_input(
                    f"Stock lengths for {label} (comma-separated, inches)",
                    value="240, 480",
                )
            run_s = st.form_submit_button("Calculate structural yield")

        if run_s:
            yield_results_by_shape = {}
            for label, parts_list in grouped_struct.items():
                stock_str = stock_len_inputs.get(label, "")
                try:
                    stock_lengths = sorted(
                        [float(s.strip()) for s in stock_str.split(",") if s.strip() and float(s.strip()) > 0],
                        reverse=True,
                    )
                except Exception:
                    yield_results_by_shape[label] = {"error": "Invalid stock lengths."}
                    continue

                all_cuts = []
                total_req = 0.0
                for pe in parts_list:
                    all_cuts.extend([float(pe["length"])] * int(pe["quantity"]))
                    total_req += float(pe["length"]) * int(pe["quantity"])
                if not all_cuts:
                    yield_results_by_shape[label] = {"info": "No cuts required."}
                    continue

                options = []
                best = {"stock_length": None, "bars_used": float("inf"), "total_waste": float("inf"), "yield_percentage": 0.0}
                for stock_len in stock_lengths:
                    cuts_fit = [c for c in all_cuts if c <= stock_len]
                    if not cuts_fit:
                        options.append({"stock_length": stock_len, "bars_used": 0, "total_waste": 0.0, "yield_percentage": 0.0, "info": "No cuts fit."})
                        continue
                    bars, waste = logic.calculate_yield_for_stock_size(list(cuts_fit), stock_len)
                    actual_cut = sum(cuts_fit)
                    total_stock = bars * stock_len
                    yield_pct = (actual_cut / total_stock) * 100 if total_stock > 0 else 0.0
                    can_make_all = len(cuts_fit) == len(all_cuts)
                    current = {
                        "stock_length": stock_len,
                        "bars_used": bars,
                        "total_waste": round(waste, 2),
                        "yield_percentage": round(yield_pct, 2),
                        "can_make_all_parts": can_make_all,
                    }
                    options.append(current)
                    if can_make_all:
                        if (current["bars_used"] < best["bars_used"]) or (
                            current["bars_used"] == best["bars_used"] and current["total_waste"] < best["total_waste"]
                        ):
                            best = current.copy()
                for o in options:
                    o["is_best"] = best["stock_length"] is not None and o.get("stock_length") == best.get("stock_length")
                yield_results_by_shape[label] = {
                    "options": options,
                    "best_overall": best if best["stock_length"] is not None else None,
                    "total_required_length": round(total_req, 2),
                }
            st.session_state["structural_yield_results"] = yield_results_by_shape

        if st.session_state["structural_yield_results"]:
            for label, res in st.session_state["structural_yield_results"].items():
                st.markdown(f"**{label}**")
                if "error" in res:
                    st.error(res["error"])
                    continue
                if "info" in res:
                    st.info(res["info"])
                    continue
                st.write({"Total required length (in)": res.get("total_required_length")})
                for o in res.get("options", []):
                    tag = "✅ best" if o.get("is_best") else ""
                    st.write(
                        f"{tag} Stock {o['stock_length']:.2f} in → bars: {o['bars_used']}, "
                        f"waste: {o.get('total_waste',0):.2f} in, yield: {o.get('yield_percentage',0):.2f}% "
                        f"({'ALL parts' if o.get('can_make_all_parts') else 'partial'})"
                    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_state()
    require_auth()

    # Load AISC DB
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
