import io
import os
import hmac
import datetime
import math
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

APP_TITLE = "Estimating Calculator(V3)"


def _init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("estimate_parts", [])
    st.session_state.setdefault("plate_yield_results", {})
    st.session_state.setdefault("structural_yield_results", {})

    # Plate nesting editor persistence
    st.session_state.setdefault("plate_stock_df", pd.DataFrame([
        {"Width (in)": 96.0, "Height (in)": 240.0, "Qty (optional)": ""},
        {"Width (in)": 120.0, "Height (in)": 240.0, "Qty (optional)": ""},
    ]))

    # Plate STEP-derived defaults (used to populate plate form)
    st.session_state.setdefault("plate_part_name", "Unnamed Plate")
    st.session_state.setdefault("plate_qty", 1)
    st.session_state.setdefault("plate_w", 0.0)
    st.session_state.setdefault("plate_l", 0.0)
    st.session_state.setdefault("plate_step_volume_in3", 0.0)
    st.session_state.setdefault("plate_step_weight_lbs", 0.0)
    st.session_state.setdefault("plate_step_bbox_h_in", 0.0)
    st.session_state.setdefault("plate_step_loaded_name", "")

    # Persist STEP parts table edits so checkbox changes stick across reruns
    st.session_state.setdefault("step_parts_df", None)
    st.session_state.setdefault("step_parts_signature", None)


def _get_password_from_secrets_or_env() -> str:
    """Return app password from Streamlit secrets (preferred) or env var as fallback."""
    if "auth" in st.secrets and "password" in st.secrets["auth"]:
        return str(st.secrets["auth"]["password"])
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
        "Rolling Required",
        "Rolling Type",
        "Rolling OD Bucket",
        "Rolling Prebend",
        "Rolling Tight Tolerance",
        "Rolling Time (min/item)",
        "Total Rolling Run Time (min)",
        "Bends (per item)",
        "Bend Complexity",
        "Perimeter (in/item)",
        "End Perimeter Both Ends (in/item)",
        "Total End Perimeter Both Ends (in)",
        "Feedrate (IPM)",
        "Weight per Foot (lbs/ft)",
        "Cross-sectional Area (in^2)",
        "Net Weight (lbs/item)",
        "Gross Weight (lbs/item)",
        "Burning Time (min/item)",
        "Drilling Time (min/item)",
        "Bend Time (min/item)",
        "Cutting Time (min/item)",
        "Fit Time (min/item)",
        "Total Gross Weight (lbs)",
        "Total Burning Time (min)",
        "Total Drilling Time (min)",
        "Total Bend Time (min)",
        "Total Cutting Time (min)",
        "Total Fit Time (min)",
        "Total Weld Wire (lbs)",
        "Total Weld Time (hours)",
        "STEP Volume (in^3)",
        "STEP Weight (lbs/item)",
        "STEP BBox H (in)",
        "STEP Source File",
        "DXF Source",
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


def _structural_end_perimeter_one_end_in(props: Dict[str, Any]) -> float:
    """Approximate end perimeter (inches) for ONE end of a structural shape from AISC props."""

    def _get_float(*keys: str) -> float:
        for k in keys:
            if not k:
                continue
            for kk in (k, f"{k}_float"):
                if kk in props and props.get(kk) not in (None, ""):
                    try:
                        return float(props.get(kk))
                    except Exception:
                        pass
            lk = k.lower()
            for pk, pv in props.items():
                if str(pk).lower() == lk and pv not in (None, ""):
                    try:
                        return float(pv)
                    except Exception:
                        pass
        return 0.0

    shape = str(props.get("Shape", "") or props.get("Type", "") or "").upper()
    label = str(props.get("EDI_Std_Nomenclature", "") or props.get("Label", "") or "").upper()
    blob = f"{shape} {label}"

    if "HSS" in blob:
        b = _get_float("B", "b")
        h = _get_float("Ht", "H", "h")
        if b > 0 and h > 0:
            return 2.0 * (b + h)

        od = _get_float("OD", "ODt", "D", "d")
        if od > 0:
            return math.pi * od

    if "PIPE" in blob or "TUBE" in blob:
        od = _get_float("OD", "ODt", "D", "d")
        if od > 0:
            return math.pi * od

    bf = _get_float("bf", "BF")
    d = _get_float("d", "D")
    if bf > 0 and d > 0:
        return 2.0 * (bf + d)

    a = _get_float("A", "A_float")
    if a > 0:
        return 4.0 * math.sqrt(a)

    return 0.0


# ============================================================
# Setup time helpers (NEW)
# ============================================================

SETUP_MINUTES_PER_UNIQUE_MAT_THK = 30.0


def _safe_setup_key(material: Any, thickness: Any) -> Tuple[str, float]:
    """Normalize material/thickness into a stable hashable key."""
    mat = "" if material is None else str(material).strip()
    try:
        thk = float(thickness)
    except Exception:
        thk = 0.0
    return (mat, thk)


def _calculate_setup_times(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate setup times (minutes), separate from runtime:
    - Laser: Plate items whose Burn Machine Type == "Laser"
    - Kinetic: Plate items whose Burn Machine Type == "Kinetic"
    - Saw: Structural items (cutting)
    Rule: 30 minutes per unique (Material, Thickness) per process.
    """

    laser_keys = set()
    kinetic_keys = set()
    saw_keys = set()

    for r in rows:
        etype = r.get("Estimation Type")
        if etype == "Plate":
            key = _safe_setup_key(r.get("Material"), r.get("Thickness (in)"))
            burn_type = r.get("Burn Machine Type")
            if burn_type == "Laser":
                laser_keys.add(key)
            elif burn_type == "Kinetic":
                kinetic_keys.add(key)

        elif etype == "Structural":
            # Structural doesn't currently store Material/Thickness; use a single setup bucket.
            # If you add struct material/grade later, we can key off that too.
            saw_keys.add(("STRUCTURAL", 0.0))

    return {
        "laser_setup_min": float(len(laser_keys)) * SETUP_MINUTES_PER_UNIQUE_MAT_THK,
        "kinetic_setup_min": float(len(kinetic_keys)) * SETUP_MINUTES_PER_UNIQUE_MAT_THK,
        "saw_setup_min": float(len(saw_keys)) * SETUP_MINUTES_PER_UNIQUE_MAT_THK,
        "laser_setup_count": float(len(laser_keys)),
        "kinetic_setup_count": float(len(kinetic_keys)),
        "saw_setup_count": float(len(saw_keys)),
    }


# ============================================================
# Cone development helpers + shared rolling heuristic
# ============================================================

def _rolling_time_minutes_per_item(
    weight_lbs: float,
    rolling_type: str,
    od_bucket: str,
    prebend: bool,
    tight_tolerance: bool,
) -> float:
    """Heuristic rolling runtime (minutes/item) based on part weight + options."""
    w = max(0.0, float(weight_lbs or 0.0))

    if w <= 250:
        base = 15
    elif w <= 500:
        base = 30
    elif w <= 1000:
        base = 45
    elif w <= 2000:
        base = 75
    elif w <= 4000:
        base = 120
    else:
        base = 180

    rtype = (rolling_type or "Cylinder").strip().lower()
    if rtype.startswith("cone"):
        base *= 1.25

    bucket = (od_bucket or "").lower()
    if "small" in bucket or "<" in bucket:
        base *= 1.20
    elif "medium" in bucket:
        base *= 1.10

    if prebend:
        base += 15
    if tight_tolerance:
        base += 30

    return float(round(base, 2))


def _cone_development_truncated(D1: float, D2: float, H: float, use_mean_diam: bool, thickness: float) -> Dict[str, float]:
    """Truncated cone development as an annular sector."""
    if D1 <= 0 or D2 < 0 or H <= 0:
        raise ValueError("D1 must be > 0, D2 must be >= 0, H must be > 0.")
    r_large = D1 / 2.0
    r_small = D2 / 2.0
    if use_mean_diam:
        r_large = max(0.0, r_large - thickness / 2.0)
        r_small = max(0.0, r_small - thickness / 2.0)

    if r_large <= r_small:
        raise ValueError("Large diameter must be greater than small diameter.")

    H_full = H * (r_large / (r_large - r_small))
    slant_large = math.sqrt(H_full**2 + r_large**2)
    slant_small = math.sqrt((H_full - H)**2 + r_small**2)

    theta_rad = (2.0 * math.pi * r_large) / slant_large
    theta_deg = math.degrees(theta_rad)

    arc_out = theta_rad * slant_large
    arc_in = theta_rad * slant_small

    return {
        "r_large": r_large,
        "r_small": r_small,
        "H_full": H_full,
        "Rout": slant_large,
        "Rin": slant_small,
        "theta_rad": theta_rad,
        "theta_deg": theta_deg,
        "arc_out": arc_out,
        "arc_in": arc_in,
        "slant_large": slant_large,
        "slant_small": slant_small,
    }


def _cone_sector_area(Rout: float, Rin: float, theta_rad: float) -> float:
    """Area of an annular sector."""
    return 0.5 * float(theta_rad) * (float(Rout) ** 2 - float(Rin) ** 2)


def page_cone_calculator() -> None:
    st.header("Cone Calculator")
    st.caption(
        "Build a truncated cone flat pattern (annular sector). "
        "You can add the developed gore(s) into the estimate as Plate line items."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        D1 = st.number_input('Large Diameter D1 (in)', min_value=0.0, value=60.0, step=0.25, key="cone_D1")
        D2 = st.number_input('Small Diameter D2 (in)', min_value=0.0, value=30.0, step=0.25, key="cone_D2")
        H = st.number_input('Height H (in)', min_value=0.0, value=24.0, step=0.25, key="cone_H")
        gores = st.number_input('Number of gores (pieces)', min_value=1, value=1, step=1, key="cone_gores")
    with c2:
        material = st.selectbox("Material", options=logic.MATERIALS_LIST, index=0, key="cone_mat")
        thickness = st.selectbox("Thickness (in)", options=logic.THICKNESS_LIST, index=0, key="cone_thk")
        quantity = st.number_input("Quantity (per gore)", min_value=1, value=1, step=1, key="cone_qty")
        use_mean = st.checkbox("Use mean diameter (OD - t)", value=True, key="cone_use_mean")
    with c3:
        seam_allow = st.number_input(
            "Seam allowance per radial edge (in)",
            min_value=0.0,
            value=0.5,
            step=0.125,
            help="Adds extra length to each radial edge for fit-up/trim.",
            key="cone_seam_allow",
        )
        add_roll = st.checkbox("Rolling required (estimate rolling time)", value=True, key="cone_roll_required")
        rolling_od_bucket = st.selectbox(
            "OD bucket",
            ["Small OD (<24\")", "Medium OD (24–60\")", "Large OD (60\"+)"],
            index=2,
            key="cone_roll_od",
        )
        rolling_prebend = st.checkbox("Prebend", value=False, key="cone_roll_prebend")
        rolling_tight_tol = st.checkbox("Tight tolerance", value=False, key="cone_roll_tighttol")

    try:
        dev = _cone_development_truncated(
            D1=float(D1),
            D2=float(D2),
            H=float(H),
            use_mean_diam=bool(use_mean),
            thickness=float(thickness),
        )
    except Exception as e:
        st.error(f"Cannot compute cone development: {e}")
        return

    Rout = float(dev["Rout"])
    Rin = float(dev["Rin"])
    theta_rad = float(dev["theta_rad"])
    theta_deg = float(dev["theta_deg"])

    n_gores = int(gores)
    theta_g = theta_rad / n_gores
    theta_g_deg = math.degrees(theta_g)

    arc_out_g = dev["arc_out"] / n_gores
    arc_in_g = dev["arc_in"] / n_gores

    radial_edge = (Rout - Rin) + float(seam_allow)
    cut_perim_g = float(arc_out_g) + float(arc_in_g) + 2.0 * float(radial_edge)

    area_total = _cone_sector_area(Rout, Rin, theta_rad)
    area_g = area_total / n_gores

    density = float(getattr(logic, "DENSITY_FACTOR_FOR_CALCULATION", 0.283))
    net_weight_g = float(area_g) * float(thickness) * float(density)

    gross_weight_g = logic.calculate_gross_weight(net_weight_g, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT)
    fit_time_g = logic.calculate_fit_time(net_weight_g)

    burn_machine = logic.get_plate_burn_machine_type(thickness)
    feedrate = logic.get_feedrate_for_thickness(thickness, logic.FEEDRATE_TABLE_IPM)
    burn_time_g = round(logic.calculate_burning_time(cut_perim_g, feedrate), 2)

    rolling_time_item = 0.0
    total_rolling_time = 0.0
    if add_roll:
        rolling_time_item = _rolling_time_minutes_per_item(
            weight_lbs=net_weight_g,
            rolling_type="Cone",
            od_bucket=rolling_od_bucket,
            prebend=rolling_prebend,
            tight_tolerance=rolling_tight_tol,
        )
        total_rolling_time = round(float(rolling_time_item) * int(quantity), 2)

    st.subheader("Development results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Outer radius (Rout)", f"{Rout:.3f}")
    m2.metric("Inner radius (Rin)", f"{Rin:.3f}")
    m3.metric("Included angle (θ)", f"{theta_deg:.3f}°")
    m4.metric("Angle per gore", f"{theta_g_deg:.3f}°")

    st.write("**Per gore:**")
    st.write(f"- Outer arc: **{arc_out_g:.3f} in**")
    st.write(f"- Inner arc: **{arc_in_g:.3f} in**")
    st.write(f"- Radial edge (incl seam allowance): **{radial_edge:.3f} in**")
    st.write(f"- Cut perimeter: **{cut_perim_g:.3f} in**")
    st.write(f"- Developed area: **{area_g:.3f} in²**")
    st.write(f"- Net weight (est.): **{net_weight_g:.2f} lb**")

    st.divider()

    st.subheader("Add to estimate as Plate")
    part_name = st.text_input(
        "Part name (gore)",
        value=f"Cone gore (D1={float(D1):.1f}, D2={float(D2):.1f}, H={float(H):.1f})",
        key="cone_part_name",
    )

    add_btn = st.button("Add cone gore(s) to estimate", type="primary", key="cone_add_btn")
    if add_btn:
        notes = (
            f"Cone dev | D1={float(D1):.3f}, D2={float(D2):.3f}, H={float(H):.3f}, "
            f"Rout={Rout:.3f}, Rin={Rin:.3f}, θ_total={theta_deg:.3f}°, gores={n_gores}, "
            f"θ_gore={theta_g_deg:.3f}°, seam_allow={float(seam_allow):.3f}"
        )

        part = {
            "Estimation Type": "Plate",
            "Part Name": str(part_name),
            "Quantity": int(quantity),
            "Material": material,
            "Thickness (in)": float(thickness),
            "Width (in)": round(float(arc_out_g), 3),
            "Length (in)": round(float(radial_edge), 3),
            "Bends (per item)": 0,
            "Bend Complexity": "N/A",
            "Burn Machine Type": burn_machine,
            "Perimeter (in/item)": round(float(cut_perim_g), 2),
            "Feedrate (IPM)": float(feedrate),
            "Drilling Time (min/item)": 0.0,
            "Drill Details Summary": "",
            "Burning Time (min/item)": float(burn_time_g),
            "Bend Time (min/item)": 0.0,
            "Rolling Required": "Yes" if add_roll else "No",
            "Rolling Type": "Cone" if add_roll else "",
            "Rolling OD Bucket": rolling_od_bucket if add_roll else "",
            "Rolling Prebend": "Yes" if (add_roll and rolling_prebend) else "No" if add_roll else "",
            "Rolling Tight Tolerance": "Yes" if (add_roll and rolling_tight_tol) else "No" if add_roll else "",
            "Rolling Time (min/item)": float(rolling_time_item),
            "Net Weight (lbs/item)": round(float(net_weight_g), 2),
            "Gross Weight (lbs/item)": round(float(gross_weight_g), 2),
            "Fit Time (min/item)": float(fit_time_g),
            "Total Gross Weight (lbs)": round(float(gross_weight_g) * int(quantity), 2),
            "Total Burning Time (min)": round(float(burn_time_g) * int(quantity), 2),
            "Total Drillingilling Time (min)": 0.0,
            "Total Drilling Time (min)": 0.0,
            "Total Bend Time (min)": 0.0,
            "Total Rolling Run Time (min)": float(total_rolling_time),
            "Total Fit Time (min)": round(float(fit_time_g) * int(quantity), 2),
            "Cone D1 (in)": float(D1),
            "Cone D2 (in)": float(D2),
            "Cone H (in)": float(H),
            "Cone Rout (in)": float(Rout),
            "Cone Rin (in)": float(Rin),
            "Cone Theta Total (deg)": float(theta_deg),
            "Cone Theta Gore (deg)": float(theta_g_deg),
            "Cone Seam Allow (in)": float(seam_allow),
            "Cone Notes": notes,
        }

        _add_part(part)
        st.success("Cone gore(s) added as Plate line item(s).")
        st.rerun()


def page_plate() -> None:
    st.header("Plate")

    # ------------------------------------------------------------
    # 3D STEP Import (BBOX / Volume / Weight)
    # ------------------------------------------------------------
    with st.expander("3D STEP Import (BBOX / Volume / Weight)", expanded=False):
        st.caption(
            "Upload .STEP/.STP files to extract overall bounding box and volume/weight. "
            "Smallest bbox dimension is treated as thickness; the other two become width/length."
        )

        u1, u2, u3 = st.columns([1.2, 1, 2])
        with u1:
            geom_units = st.selectbox(
                "Imported geometry units",
                options=["meters (recommended)", "millimeters", "inches"],
                index=0,
                help="Most STEP loads via trimesh+cascadio come in meters. If bbox/volume looks off, switch this.",
                key="step_units",
            )
        with u2:
            step_scale = st.number_input(
                "Scale multiplier",
                min_value=0.0001,
                value=1.0,
                step=0.1,
                help="Applied after unit conversion. Example: if model is 1/10 scale, enter 10.0.",
                key="step_scale",
            )
        with u3:
            density = st.number_input(
                "Density (lb / in³)",
                min_value=0.0001,
                value=float(getattr(logic, "DENSITY_FACTOR_FOR_CALCULATION", 0.283)),
                step=0.001,
                help="Steel is ~0.283 lb/in³. Stainless is ~0.289 lb/in³.",
                key="step_density",
            )

        step_files = st.file_uploader(
            "Upload STEP file(s) (.step/.stp)",
            type=["step", "stp", "STEP", "STP"],
            accept_multiple_files=True,
            key="step_upload_multi",
        )

        def _units_to_inches_factor(units_label: str) -> float:
            label = (units_label or "").lower()
            if "meter" in label:
                return 39.37007874015748
            if "millimeter" in label:
                return 1.0 / 25.4
            return 1.0

        def _mesh_preview_plotly(mesh, max_faces: int = 8000):
            import plotly.graph_objects as go

            m = mesh
            try:
                face_count = int(len(getattr(m, "faces", [])))
                if face_count > max_faces:
                    if hasattr(m, "simplify_quadratic_decimation"):
                        m = m.simplify_quadratic_decimation(max_faces)
            except Exception:
                pass

            v = m.vertices
            f = m.faces

            fig = go.Figure(
                data=[
                    go.Mesh3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        opacity=1.0,
                    )
                ]
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(
                    aspectmode="data",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                showlegend=False,
            )
            return fig

        def _extract_step_parts_table(step_bytes: bytes, filename: str, units_label: str, scale: float, density_lb_in3: float):
            import trimesh
            import uuid

            scene_or_mesh = trimesh.load(file_obj=io.BytesIO(step_bytes), file_type="step")
            to_in = _units_to_inches_factor(units_label)
            factor = float(to_in) * float(scale)

            def _metrics(m):
                ext_raw = m.extents
                dims_in = [
                    float(ext_raw[0] * factor),
                    float(ext_raw[1] * factor),
                    float(ext_raw[2] * factor),
                ]
                dims_sorted = sorted(dims_in)
                t_in = float(dims_sorted[0])
                w_in = float(dims_sorted[1])
                l_in = float(dims_sorted[2])

                vol_raw = float(m.volume)
                vol_in3 = vol_raw * (factor ** 3)

                if not (vol_in3 > 0):
                    try:
                        vol_raw2 = float(m.convex_hull.volume)
                        vol_in3 = vol_raw2 * (factor ** 3)
                    except Exception:
                        vol_in3 = 0.0

                wt_lb = float(vol_in3) * float(density_lb_in3)
                return w_in, l_in, t_in, float(vol_in3), float(wt_lb)

            rows = []
            meshes_by_id = {}

            def _snap_thickness_to_list(t_in: float) -> float:
                try:
                    opts = [float(x) for x in logic.THICKNESS_LIST]
                    if not opts:
                        return float(t_in)
                    return float(min(opts, key=lambda x: abs(x - float(t_in))))
                except Exception:
                    return float(t_in)

            if isinstance(scene_or_mesh, trimesh.Scene):
                for gname, geom in scene_or_mesh.geometry.items():
                    if not isinstance(geom, trimesh.Trimesh):
                        continue
                    row_id = str(uuid.uuid4())[:8]
                    bw, bl, bt, vol, wt = _metrics(geom)
                    rows.append(
                        {
                            "Include": True,
                            "Source File": filename,
                            "Part Name": str(gname) if gname else "Part",
                            "Row ID": row_id,
                            "Qty": 1,
                            "Material": str(logic.MATERIALS_LIST[0]),
                            "Thickness (in)": _snap_thickness_to_list(bt),
                            "Use STEP weight": True,
                            "BBox W (in)": round(bw, 3),
                            "BBox L (in)": round(bl, 3),
                            "Inferred Thickness (in)": round(bt, 4),
                            "Volume (in³)": round(vol, 3),
                            "Weight (lb)": round(wt, 2),
                        }
                    )
                    meshes_by_id[row_id] = geom
            elif hasattr(scene_or_mesh, "vertices") and hasattr(scene_or_mesh, "faces"):
                row_id = str(uuid.uuid4())[:8]
                bw, bl, bt, vol, wt = _metrics(scene_or_mesh)
                rows.append(
                    {
                        "Include": True,
                        "Source File": filename,
                        "Part Name": os.path.splitext(filename)[0],
                        "Row ID": row_id,
                        "Qty": 1,
                        "Material": str(logic.MATERIALS_LIST[0]),
                        "Thickness (in)": _snap_thickness_to_list(bt),
                        "Use STEP weight": True,
                        "BBox W (in)": round(bw, 3),
                        "BBox L (in)": round(bl, 3),
                        "Inferred Thickness (in)": round(bt, 4),
                        "Volume (in³)": round(vol, 3),
                        "Weight (lb)": round(wt, 2),
                    }
                )
                meshes_by_id[row_id] = scene_or_mesh
            else:
                raise ValueError("Unsupported geometry returned from STEP loader.")

            return pd.DataFrame(rows), meshes_by_id

        if step_files:
            all_rows = []
            mesh_cache = {}

            for f in step_files:
                try:
                    df_parts_tmp, meshes_by_id = _extract_step_parts_table(
                        step_bytes=f.getvalue(),
                        filename=f.name,
                        units_label=geom_units,
                        scale=float(step_scale),
                        density_lb_in3=float(density),
                    )
                    all_rows.extend(df_parts_tmp.to_dict("records"))
                    mesh_cache.update(meshes_by_id)
                except Exception as e:
                    st.error(
                        "STEP import failed. On Streamlit Cloud include dependencies "
                        "`trimesh` and `cascadio` in requirements.txt (and `plotly` for preview). Error: "
                        + str(e)
                    )

            if not all_rows:
                st.warning("No parts found in the uploaded STEP file(s).")
            else:
                sig = (
                    tuple((f.name, len(f.getvalue())) for f in step_files),
                    str(geom_units),
                    float(step_scale),
                    float(density),
                )

                if (st.session_state.get("step_parts_signature") != sig) or (st.session_state.get("step_parts_df") is None):
                    df_parts = pd.DataFrame(all_rows)
                    for col in ["Include", "Use STEP weight"]:
                        if col in df_parts.columns:
                            try:
                                df_parts[col] = df_parts[col].astype(bool)
                            except Exception:
                                pass
                    st.session_state["step_parts_df"] = df_parts
                    st.session_state["step_parts_signature"] = sig

                st.markdown("#### STEP parts")
                st.caption(
                    "Edit Qty / Material / Thickness. Uncheck 'Use STEP weight' if you want to use plate-calculated weight instead."
                )

                df_parts = st.session_state["step_parts_df"]

                edited = st.data_editor(
                    df_parts,
                    use_container_width=True,
                    num_rows="fixed",
                    key="step_parts_editor_multi",
                    column_config={
                        "Include": st.column_config.CheckboxColumn("Include", default=True),
                        "Qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, required=False),
                        "Material": st.column_config.SelectboxColumn(
                            "Material", options=[str(x) for x in logic.MATERIALS_LIST], required=True
                        ),
                        "Thickness (in)": st.column_config.SelectboxColumn(
                            "Thickness (in)", options=[float(x) for x in logic.THICKNESS_LIST], required=True
                        ),
                        "Use STEP weight": st.column_config.CheckboxColumn("Use STEP weight", default=True),
                        "BBox W (in)": st.column_config.NumberColumn("BBox W (in)", disabled=True),
                        "BBox L (in)": st.column_config.NumberColumn("BBox L (in)", disabled=True),
                        "Inferred Thickness (in)": st.column_config.NumberColumn("Inferred Thickness (in)", disabled=True),
                        "Volume (in³)": st.column_config.NumberColumn("Volume (in³)", disabled=True),
                        "Weight (lb)": st.column_config.NumberColumn("Weight (lb)", disabled=True),
                        "Row ID": st.column_config.TextColumn("Row ID", disabled=True),
                    },
                )

                st.session_state["step_parts_df"] = edited

                st.markdown("**Load STEP row into Plate inputs (below)**")
                try:
                    if not edited.empty:
                        labels = [
                            f"{r['Source File']} :: {r['Part Name']} (id={r['Row ID']})"
                            for _, r in edited.iterrows()
                        ]
                        sel = st.selectbox("Select a STEP row to load", options=labels, key="step_load_select")
                        if st.button("Load selected STEP row into Plate form", key="step_load_to_plate_btn"):
                            row_id = sel.split("id=")[-1].rstrip(")").strip()
                            row = edited[edited["Row ID"] == row_id].iloc[0].to_dict()

                            st.session_state["plate_part_name"] = str(row.get("Part Name", "STEP Part"))
                            st.session_state["plate_qty"] = int(row.get("Qty", 1) or 1)
                            st.session_state["plate_w"] = float(row.get("BBox W (in)", 0.0) or 0.0)
                            st.session_state["plate_l"] = float(row.get("BBox L (in)", 0.0) or 0.0)
                            st.session_state["plate_step_volume_in3"] = float(row.get("Volume (in³)", 0.0) or 0.0)
                            st.session_state["plate_step_weight_lbs"] = float(row.get("Weight (lb)", 0.0) or 0.0)
                            st.session_state["plate_step_bbox_h_in"] = float(row.get("Inferred Thickness (in)", 0.0) or 0.0)
                            st.session_state["plate_step_loaded_name"] = str(row.get("Source File", ""))

                            try:
                                st.session_state["plate_mat"] = str(row.get("Material", logic.MATERIALS_LIST[0]))
                            except Exception:
                                pass
                            try:
                                st.session_state["plate_thk"] = float(row.get("Thickness (in)", logic.THICKNESS_LIST[0]))
                            except Exception:
                                pass

                            st.success("Loaded STEP row into Plate inputs below. Add drilling/rolling and click 'Add plate'.")
                            st.rerun()
                except Exception:
                    pass

                with st.expander("Preview a selected part (optional)", expanded=False):
                    try:
                        selectable = edited[edited["Include"] == True].copy()  # noqa: E712
                        if selectable.empty:
                            st.info("Select at least one part (Include = true) to preview.")
                        else:
                            labels2 = [
                                f"{r['Source File']} :: {r['Part Name']} (id={r['Row ID']})"
                                for _, r in selectable.iterrows()
                            ]
                            sel2 = st.selectbox("Choose a part to preview", options=labels2, key="step_preview_select_multi")
                            row_id2 = sel2.split("id=")[-1].rstrip(")").strip()
                            mesh = mesh_cache.get(row_id2)
                            if mesh is None:
                                st.warning("Preview mesh not available for this part.")
                            else:
                                fig = _mesh_preview_plotly(mesh)
                                st.plotly_chart(fig, use_container_width=True, height=320)
                    except Exception as e:
                        st.warning(f"Preview failed: {e}")

                add_selected = st.button(
                    "Add selected STEP parts to estimate (no drilling/rolling)",
                    type="secondary",
                    key="step_add_selected_multi",
                )
                if add_selected:
                    added_n = 0
                    for _, r in edited.iterrows():
                        try:
                            if not bool(r.get("Include", True)):
                                continue

                            qty = int(r.get("Qty", 1))
                            material = str(r.get("Material", logic.MATERIALS_LIST[0]))
                            thickness = float(r.get("Thickness (in)", logic.THICKNESS_LIST[0]))

                            width = float(r.get("BBox W (in)", 0.0))
                            length = float(r.get("BBox L (in)", 0.0))
                            perimeter = float(2.0 * (width + length))

                            burn_machine = logic.get_plate_burn_machine_type(thickness)
                            feedrate = logic.get_feedrate_for_thickness(thickness, logic.FEEDRATE_TABLE_IPM)

                            drilling_time_item = 0.0
                            drill_summary_str = ""
                            burn_time_item = round(logic.calculate_burning_time(perimeter, feedrate), 2)
                            bend_time_item = 0.0

                            net_weight_item = logic.calculate_plate_net_weight(
                                thickness, width, length, getattr(logic, "DENSITY_FACTOR_FOR_CALCULATION", 0.283)
                            )
                            use_step_weight = bool(r.get("Use STEP weight", True))
                            step_weight = float(r.get("Weight (lb)", 0.0) or 0.0)
                            step_volume = float(r.get("Volume (in³)", 0.0) or 0.0)
                            bbox_h = float(r.get("Inferred Thickness (in)", 0.0) or 0.0)
                            source_file = str(r.get("Source File", ""))
                            part_name = str(r.get("Part Name", "STEP Part"))

                            if use_step_weight and step_weight > 0:
                                net_weight_item = step_weight

                            gross_weight_item = logic.calculate_gross_weight(net_weight_item, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT)
                            fit_time_item = logic.calculate_fit_time(net_weight_item)

                            part = {
                                "Estimation Type": "Plate",
                                "Part Name": part_name,
                                "Quantity": qty,
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
                                "Rolling Required": "No",
                                "Rolling Type": "",
                                "Rolling OD Bucket": "",
                                "Rolling Prebend": "",
                                "Rolling Tight Tolerance": "",
                                "Rolling Time (min/item)": 0.0,
                                "Net Weight (lbs/item)": round(net_weight_item, 2),
                                "Gross Weight (lbs/item)": round(gross_weight_item, 2),
                                "Fit Time (min/item)": float(fit_time_item),
                                "STEP Volume (in^3)": step_volume,
                                "STEP Weight (lbs/item)": step_weight,
                                "STEP BBox H (in)": bbox_h,
                                "STEP Source File": source_file,
                                "Total Gross Weight (lbs)": round(gross_weight_item * qty, 2),
                                "Total Burning Time (min)": round(burn_time_item * qty, 2),
                                "Total Drilling Time (min)": 0.0,
                                "Total Bend Time (min)": 0.0,
                                "Total Rolling Run Time (min)": 0.0,
                                "Total Fit Time (min)": round(fit_time_item * qty, 2),
                            }

                            _add_part(part)
                            added_n += 1
                        except Exception as e:
                            st.error(f"Could not add a selected STEP part: {e}")

                    if added_n:
                        st.success(f"Added {added_n} part(s) from STEP file(s).")
                        st.rerun()

    # ------------------------------------------------------------
    # DXF Batch Import (Plate Only)
    # ------------------------------------------------------------
    with st.expander("DXF Batch Import (Plate Only)", expanded=False):
        st.caption(
            "Upload one or more DXF files. We detect plate size and cut perimeter. "
            "Then you can edit thickness/material/qty and add to the estimate."
        )

        up_cols = st.columns([1, 1, 1, 2])
        with up_cols[0]:
            dxf_units = st.selectbox("DXF units", options=["Inches", "mm"], index=0, key="dxf_units")
        with up_cols[1]:
            flatten_tol = st.number_input("Curve tolerance (in)", min_value=0.001, value=0.01, step=0.001, key="dxf_flatten_tol")
        with up_cols[2]:
            dxf_scale = st.number_input(
                "Scale multiplier",
                min_value=0.0001,
                value=1.0,
                step=0.1,
                help="Applied after unit conversion. Example: if DXF geometry is at 1/8 scale, enter 8.0.",
                key="dxf_scale",
            )
        with up_cols[3]:
            ignore_layers_text = st.text_input(
                "Ignore layers containing (comma-separated)",
                value="ETCH,SCRIBE,ENGRAVE,MARK,TEXT,DIM,CENTER,CL",
                key="dxf_ignore_layers",
            )

        strict_single = st.checkbox(
            "Strict single-part mode (fail if multiple outer profiles are found)",
            value=False,
            key="dxf_strict_single",
        )

        uploaded = st.file_uploader(
            "Upload DXF files",
            type=["dxf", "DXF"],
            accept_multiple_files=True,
            key="dxf_upload_multi",
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
                            "Thickness (in)", options=[float(x) for x in logic.THICKNESS_LIST], required=True
                        ),
                        "Grade": st.column_config.SelectboxColumn(
                            "Grade", options=[str(x) for x in logic.MATERIALS_LIST], required=True
                        ),
                        "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1, required=False),
                    },
                )

                add_all = st.button("Add all detected plate parts to estimate", key="dxf_add_all")
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
                                thickness, width, length, getattr(logic, "DENSITY_FACTOR_FOR_CALCULATION", 0.283)
                            )
                            gross_weight_item = logic.calculate_gross_weight(net_weight_item, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT)
                            fit_time_item = logic.calculate_fit_time(net_weight_item)

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
                                "Rolling Required": "No",
                                "Rolling Type": "",
                                "Rolling OD Bucket": "",
                                "Rolling Prebend": "",
                                "Rolling Tight Tolerance": "",
                                "Rolling Time (min/item)": 0.0,
                                "Net Weight (lbs/item)": round(net_weight_item, 2),
                                "Gross Weight (lbs/item)": round(gross_weight_item, 2),
                                "Fit Time (min/item)": float(fit_time_item),
                                "Total Gross Weight (lbs)": round(gross_weight_item * quantity, 2),
                                "Total Burning Time (min)": round(burn_time_item * quantity, 2),
                                "Total Drilling Time (min)": 0.0,
                                "Total Bend Time (min)": 0.0,
                                "Total Rolling Run Time (min)": 0.0,
                                "Total Fit Time (min)": round(fit_time_item * quantity, 2),
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
                st.warning("No closed profiles detected in uploaded DXFs (or filtered out by ignored layers).")

    # ------------------------------------------------------------
    # Manual Plate entry (supports drilling + rolling)
    # ------------------------------------------------------------

    c1, c2 = st.columns(2)
    with c1:
        part_name = st.text_input(
            "Part name",
            value=st.session_state.get("plate_part_name", "Unnamed Plate"),
            key="plate_part_name",
        )
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=int(st.session_state.get("plate_qty", 1)),
            step=1,
            key="plate_qty",
        )
        material = st.selectbox("Material", options=logic.MATERIALS_LIST, index=0, key="plate_mat")
        thickness = st.selectbox("Thickness (in)", options=logic.THICKNESS_LIST, index=0, key="plate_thk")
    with c2:
        width = st.number_input(
            "Width (in)",
            min_value=0.0,
            value=float(st.session_state.get("plate_w", 0.0)),
            step=0.25,
            key="plate_w",
        )
        length = st.number_input(
            "Length (in)",
            min_value=0.0,
            value=float(st.session_state.get("plate_l", 0.0)),
            step=0.25,
            key="plate_l",
        )
        num_bends = st.number_input("Bends (per item)", min_value=0, value=0, step=1)
        bend_complexity = st.selectbox("Bend complexity", options=["N/A"] + logic.BEND_COMPLEXITY_OPTIONS, index=0)
        if num_bends == 0:
            bend_complexity = "N/A"

    if st.session_state.get("plate_step_loaded_name"):
        st.info(
            f"STEP loaded: {st.session_state.get('plate_step_loaded_name')} | "
            f"Volume: {st.session_state.get('plate_step_volume_in3', 0.0):.3f} in³ | "
            f"Weight: {st.session_state.get('plate_step_weight_lbs', 0.0):.2f} lb | "
            f"BBox H: {st.session_state.get('plate_step_bbox_h_in', 0.0):.3f} in"
        )

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
    rolling_required = st.checkbox("Rolling required", value=False, key="plate_roll_required")

    rolling_type = st.session_state.get("plate_roll_type", "Cylinder")
    rolling_od_bucket = st.session_state.get("plate_roll_od", "Large OD (60\"+)")
    rolling_prebend = st.session_state.get("plate_roll_prebend", False)
    rolling_tight_tol = st.session_state.get("plate_roll_tighttol", False)

    if rolling_required:
        r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
        with r1:
            rolling_type = st.selectbox("Rolling type", ["Cylinder", "Cone"], index=0, key="plate_roll_type")
        with r2:
            rolling_od_bucket = st.selectbox(
                "OD bucket",
                ["Small OD (<24\")", "Medium OD (24–60\")", "Large OD (60\"+)"],
                index=2,
                key="plate_roll_od",
            )
        with r3:
            rolling_prebend = st.checkbox("Prebend", value=False, key="plate_roll_prebend")
        with r4:
            rolling_tight_tol = st.checkbox("Tight tolerance", value=False, key="plate_roll_tighttol")

    st.markdown("#### STEP Weight (optional)")
    use_step_weight = st.checkbox(
        "Use STEP-derived weight for fit time (and reporting)",
        value=bool(st.session_state.get("plate_step_weight_lbs", 0.0) > 0),
        help="If enabled and a STEP file was loaded, the app uses STEP weight instead of plate formula weight.",
        key="plate_use_step_weight_manual",
    )

    st.divider()

    add = st.button("Add plate to estimate", type="primary", key="plate_add_btn")

    if add:
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
            logic.calculate_bend_time(int(num_bends), bend_complexity, logic.BEND_TIME_PER_COMPLEXITY_MINUTES),
            2,
        )

        net_weight_item = logic.calculate_plate_net_weight(
            thickness, width, length, getattr(logic, "DENSITY_FACTOR_FOR_CALCULATION", 0.283)
        )

        if use_step_weight:
            step_wt = float(st.session_state.get("plate_step_weight_lbs", 0.0) or 0.0)
            if step_wt > 0:
                net_weight_item = step_wt

        gross_weight_item = logic.calculate_gross_weight(net_weight_item, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT)
        fit_time_item = logic.calculate_fit_time(net_weight_item)

        rolling_time_item = 0.0
        total_rolling_time = 0.0
        if rolling_required:
            rolling_time_item = _rolling_time_minutes_per_item(
                weight_lbs=net_weight_item,
                rolling_type=rolling_type,
                od_bucket=rolling_od_bucket,
                prebend=rolling_prebend,
                tight_tolerance=rolling_tight_tol,
            )
            total_rolling_time = round(float(rolling_time_item) * int(quantity), 2)

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
            "Rolling Required": "Yes" if rolling_required else "No",
            "Rolling Type": rolling_type if rolling_required else "",
            "Rolling OD Bucket": rolling_od_bucket if rolling_required else "",
            "Rolling Prebend": "Yes" if (rolling_required and rolling_prebend) else "No" if rolling_required else "",
            "Rolling Tight Tolerance": "Yes" if (rolling_required and rolling_tight_tol) else "No" if rolling_required else "",
            "Rolling Time (min/item)": float(rolling_time_item),
            "Net Weight (lbs/item)": round(net_weight_item, 2),
            "Gross Weight (lbs/item)": round(gross_weight_item, 2),
            "Fit Time (min/item)": float(fit_time_item),
            "STEP Volume (in^3)": float(st.session_state.get("plate_step_volume_in3", 0.0) or 0.0),
            "STEP Weight (lbs/item)": float(st.session_state.get("plate_step_weight_lbs", 0.0) or 0.0),
            "STEP BBox H (in)": float(st.session_state.get("plate_step_bbox_h_in", 0.0) or 0.0),
            "STEP Source File": str(st.session_state.get("plate_step_loaded_name", "")),
            "Total Gross Weight (lbs)": round(gross_weight_item * int(quantity), 2),
            "Total Burning Time (min)": round(burn_time_item * int(quantity), 2),
            "Total Drilling Time (min)": round(drilling_time_item * int(quantity), 2),
            "Total Bend Time (min)": round(bend_time_item * int(quantity), 2),
            "Total Rolling Run Time (min)": float(total_rolling_time),
            "Total Fit Time (min)": round(fit_time_item * int(quantity), 2),
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

    add = st.button("Add structural to estimate", key="struct_add_btn")
    if add:
        props = logic.AISC_LABEL_TO_PROPERTIES_MAP.get(shape_label, {})
        weight_per_foot = float(props.get("W_float", 0.0) or 0.0)
        area_sq_in = float(props.get("A_float", 0.0) or 0.0)

        end_perim_one = _structural_end_perimeter_one_end_in(props)
        end_perim_both = 2.0 * float(end_perim_one)

        net_wt = logic.calculate_structural_piece_weight(weight_per_foot, length_in)
        gross_wt = logic.calculate_gross_weight(net_wt, logic.PERCENTAGE_ADD_FOR_GROSS_WEIGHT)
        fit_t = logic.calculate_fit_time(net_wt)

        cut_t = logic.calculate_structural_cutting_time(
            area_sq_in,
            logic.STRUCTURAL_CUTTING_RATE_SQ_IN_PER_MIN,
            logic.STRUCTURAL_TRAVEL_TIME_PER_PIECE_MIN,
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
            "End Perimeter Both Ends (in/item)": round(end_perim_both, 2),
            "Total End Perimeter Both Ends (in)": round(end_perim_both * int(quantity), 2),
            "Weight per Foot (lbs/ft)": weight_per_foot,
            "Cross-sectional Area (in^2)": area_sq_in,
            "Net Weight (lbs/item)": round(net_wt, 2),
            "Gross Weight (lbs/item)": round(gross_wt, 2),
            "Fit Time (min/item)": float(fit_t),
            "Cutting Time (min/item)": round(cut_t, 2),
            "Total Gross Weight (lbs)": round(gross_wt * int(quantity), 2),
            "Total Cutting Time (min)": round(cut_t * int(quantity), 2),
            "Total Fit Time (min)": round(fit_t * int(quantity), 2),
        }
        _add_part(part)
        st.success("Structural added.")
        st.rerun()

    with st.expander("Structural nesting optimization (bar cutting)", expanded=False):
        st.caption(
            "Uses the Structural items currently in your estimate (their lengths × quantities) and optimizes bar usage. "
            "Supports mixing multiple stock lengths and optional quantity limits per stock length."
        )

        rows = st.session_state.get("estimate_parts", [])
        cuts_with_qty = []
        for r in rows:
            if r.get("Estimation Type") != "Structural":
                continue
            try:
                ln = float(r.get("Length (in)", 0.0) or 0.0)
                qty = int(r.get("Quantity", 0) or 0)
                if ln > 0 and qty > 0:
                    cuts_with_qty.append({"length": ln, "qty": qty, "name": str(r.get("Part Name", ""))})
            except Exception:
                continue

        if not cuts_with_qty:
            st.info("No Structural items in the estimate yet. Add structural parts above, then come back to optimize bar usage.")
        else:
            cA, cB, cC = st.columns([1.4, 1, 1])
            with cA:
                stock_text = st.text_area(
                    "Stock length options (one per line)",
                    value="480\n240\n120",
                    height=120,
                    help=(
                        "Enter inches. Formats supported:\n"
                        "• 480\n"
                        "• 480, qty=10\n"
                        "• 480, qty=10, cost=125.00\n"
                        "If qty is omitted or 0, it is treated as unlimited."
                    ),
                    key="struct_stock_options_text",
                )
            with cB:
                kerf = st.number_input(
                    "Kerf per cut (in)",
                    min_value=0.0,
                    value=0.125,
                    step=0.01,
                    help="Approximate material lost per piece cut off the bar (saw kerf).",
                    key="struct_kerf",
                )
                end_trim = st.number_input(
                    "End trim / clamp (in)",
                    min_value=0.0,
                    value=0.0,
                    step=0.25,
                    help="Reserved length per bar for clamping/trim.",
                    key="struct_end_trim",
                )
            with cC:
                objective = st.selectbox(
                    "Optimization objective",
                    options=["waste", "bars", "cost", "balanced"],
                    index=0,
                    key="struct_objective",
                )
                trials = st.number_input(
                    "Randomized trials",
                    min_value=50,
                    max_value=5000,
                    value=600,
                    step=50,
                    key="struct_trials",
                )

            def _parse_stock_options(text: str):
                opts = []
                for raw in (text or "").splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
                    try:
                        L = float(parts[0])
                    except Exception:
                        continue
                    qty = None
                    cost = None
                    for p in parts[1:]:
                        pl = p.lower().replace(" ", "")
                        if pl.startswith("qty="):
                            try:
                                qv = int(float(pl.split("=", 1)[1]))
                                qty = None if qv <= 0 else qv
                            except Exception:
                                pass
                        elif pl.startswith("cost="):
                            try:
                                cost = float(pl.split("=", 1)[1])
                            except Exception:
                                pass
                        else:
                            try:
                                if qty is None:
                                    qv = int(float(p))
                                    qty = None if qv <= 0 else qv
                            except Exception:
                                pass
                    opts.append({"length": L, "qty": qty, "cost": cost})
                return opts

            stock_options = _parse_stock_options(stock_text)

            with st.expander("Cuts being optimized", expanded=False):
                df_cuts = pd.DataFrame(
                    [{"Part": c.get("name", ""), "Length (in)": c["length"], "Qty": c["qty"]} for c in cuts_with_qty]
                )
                st.dataframe(df_cuts, use_container_width=True)

            run_opt = st.button("Run structural nesting optimization", type="primary", key="run_struct_nesting")
            if run_opt:
                sol = logic.optimize_structural_nesting_mix(
                    cuts_with_qty=[{"length": c["length"], "qty": c["qty"]} for c in cuts_with_qty],
                    stock_options=stock_options,
                    kerf=float(kerf),
                    end_trim=float(end_trim),
                    objective=str(objective),
                    n_trials=int(trials),
                    seed=13,
                )
                st.session_state["structural_yield_results"] = sol

            sol = st.session_state.get("structural_yield_results") or {}
            if sol and sol.get("bars"):
                totals = sol.get("totals", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Bars used", f"{int(totals.get('bars_used', 0) or 0)}")
                m2.metric("Total waste (in)", f"{float(totals.get('total_waste', 0.0) or 0.0):.2f}")
                m3.metric("Utilization", f"{float(totals.get('utilization', 0.0) or 0.0) * 100.0:.2f}%")
                m4.metric("Total cost", f"${float(totals.get('total_cost', 0.0) or 0.0):.2f}")

                by_stock = sol.get("by_stock", [])
                if by_stock:
                    df_by = pd.DataFrame([
                        {
                            "Stock length (in)": r.get("stock_len"),
                            "Bars": r.get("bars"),
                            "Used (in)": round(float(r.get("total_used", 0.0) or 0.0), 2),
                            "Waste (in)": round(float(r.get("total_waste", 0.0) or 0.0), 2),
                            "Utilization %": round(float(r.get("utilization", 0.0) or 0.0) * 100.0, 2),
                        }
                        for r in by_stock
                    ])
                    st.subheader("Best mix by stock length")
                    st.dataframe(df_by, use_container_width=True)

                with st.expander("Bar-by-bar cut plan", expanded=False):
                    bars = sol.get("bars", [])
                    df_bars = pd.DataFrame([
                        {
                            "Bar #": i + 1,
                            "Stock (in)": b.get("stock_len"),
                            "Cuts (in)": ", ".join([str(round(float(x), 3)) for x in b.get("cuts", [])]),
                            "Used (in)": round(float(b.get("used_len", 0.0) or 0.0), 2),
                            "Waste (in)": round(float(b.get("waste", 0.0) or 0.0), 2),
                        }
                        for i, b in enumerate(bars)
                    ])
                    st.dataframe(df_bars, use_container_width=True)

                if sol.get("infeasible_cuts"):
                    st.warning(
                        "Some cuts could not fit into any provided stock length (after trim/kerf). "
                        f"Lengths (in): {sorted([round(float(x), 3) for x in sol.get('infeasible_cuts', [])])}"
                    )


def page_welding() -> None:
    st.header("Welding")
    st.caption("Enter welds (up to 10 shown here). Adds a single welding summary line to the estimate.")

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
            "Total Burning Time (min)": 0.0,
            "Total Drilling Time (min)": 0.0,
            "Total Bend Time (min)": 0.0,
            "Total Cutting Time (min)": 0.0,
            "Total Rolling Run Time (min)": 0.0,
            "Total Fit Time (min)": 0.0,
        }
        _add_part(part)
        st.success("Welding summary added.")
        st.rerun()


def _compute_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    plate_wt = 0.0
    struct_wt = 0.0
    plt_bend_t = 0.0
    str_cut_t = 0.0
    fit_t = 0.0
    laser_burn_t = 0.0
    kinetic_burn_t = 0.0
    drill_t = 0.0
    roll_t = 0.0
    weld_time_hr = 0.0
    weld_wire_lbs = 0.0
    perimeter_total_in = 0.0
    structural_end_perimeter_total_in = 0.0

    for r in rows:
        fit_t += float(r.get("Total Fit Time (min)", 0.0) or 0.0)
        roll_t += float(r.get("Total Rolling Run Time (min)", 0.0) or 0.0)

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
            drill_t += float(r.get("Total Drilling Time (min)", 0.0) or 0.0)
            if r.get("Burn Machine Type") == "Laser":
                laser_burn_t += float(r.get("Total Burning Time (min)", 0.0) or 0.0)
            elif r.get("Burn Machine Type") == "Kinetic":
                kinetic_burn_t += float(r.get("Total Burning Time (min)", 0.0) or 0.0)

        elif etype == "Structural":
            struct_wt += float(r.get("Total Gross Weight (lbs)", 0.0) or 0.0)
            structural_end_perimeter_total_in += float(r.get("Total End Perimeter Both Ends (in)", 0.0) or 0.0)
            str_cut_t += float(r.get("Total Cutting Time (min)", 0.0) or 0.0)

        elif etype == "Welding":
            weld_time_hr += float(r.get("Total Weld Time (hours)", 0.0) or 0.0)
            weld_wire_lbs += float(r.get("Total Weld Wire (lbs)", 0.0) or 0.0)

    setup = _calculate_setup_times(rows)

    return {
        "plate_total_gross_weight": plate_wt,
        "structural_total_gross_weight": struct_wt,

        # Run times
        "grand_total_laser_burn_time": laser_burn_t,
        "grand_total_kinetic_burn_time": kinetic_burn_t,
        "grand_total_plate_drilling_time": drill_t,
        "grand_total_plate_bend_time": plt_bend_t,
        "grand_total_structural_cutting_time": str_cut_t,
        "grand_total_fit_time": fit_t,
        "grand_total_roll_time": roll_t,

        # Setup times (NEW)
        "laser_setup_time_min": setup["laser_setup_min"],
        "kinetic_setup_time_min": setup["kinetic_setup_min"],
        "saw_setup_time_min": setup["saw_setup_min"],
        "laser_setup_count": setup["laser_setup_count"],
        "kinetic_setup_count": setup["kinetic_setup_count"],
        "saw_setup_count": setup["saw_setup_count"],

        # Totals per process (run + setup)
        "laser_total_time_min": laser_burn_t + setup["laser_setup_min"],
        "kinetic_total_time_min": kinetic_burn_t + setup["kinetic_setup_min"],
        "saw_total_time_min": str_cut_t + setup["saw_setup_min"],

        # Welding
        "grand_total_weld_time_hours": weld_time_hr,
        "grand_total_weld_wire_lbs": weld_wire_lbs,

        # Perimeter
        "grand_total_plate_perimeter_in": perimeter_total_in,
        "grand_total_structural_end_perimeter_in": structural_end_perimeter_total_in,
        "grand_total_combined_perimeter_in": perimeter_total_in + structural_end_perimeter_total_in,

        "combined_overall_gross_weight": plate_wt + struct_wt,
    }


def page_summary() -> None:
    st.header("Summary")
    rows = st.session_state["estimate_parts"]
    if not rows:
        st.info("No parts yet. Add items in Plate / Structural / Welding.")
        return

    totals = _compute_totals(rows)

    # Keep the top row exactly the same look/ordering as before
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total gross weight (lbs)", f"{totals['combined_overall_gross_weight']:.2f}")
    c2.metric("Total fit time (min)", f"{totals['grand_total_fit_time']:.2f}")
    c3.metric("Roll time (min)", f"{totals['grand_total_roll_time']:.2f}")
    c4.metric("Weld time (hr)", f"{totals['grand_total_weld_time_hours']:.2f}")

    # NEW: Separate Plate vs Structural gross weight (added under the existing top row)
    w1, w2 = st.columns(2)
    w1.metric("Plate gross weight (lbs)", f"{totals['plate_total_gross_weight']:.2f}")
    w2.metric("Structural gross weight (lbs)", f"{totals['structural_total_gross_weight']:.2f}")

    p1, p2, p3 = st.columns(3)
    p1.metric("Plate cut perimeter (in)", f"{totals['grand_total_plate_perimeter_in']:.2f}")
    p2.metric("Structural end perimeter (in)", f"{totals['grand_total_structural_end_perimeter_in']:.2f}")
    p3.metric("Combined perimeter (in)", f"{totals['grand_total_combined_perimeter_in']:.2f}")

    st.subheader("Cutting time breakdown (run vs setup)")

    l1, l2, l3 = st.columns(3)
    l1.metric("Laser run (min)", f"{totals['grand_total_laser_burn_time']:.2f}")
    l2.metric("Laser setup (min)", f"{totals['laser_setup_time_min']:.2f}")
    l3.metric("Laser total (min)", f"{totals['laser_total_time_min']:.2f}")

    k1, k2, k3 = st.columns(3)
    k1.metric("Kinetic run (min)", f"{totals['grand_total_kinetic_burn_time']:.2f}")
    k2.metric("Kinetic setup (min)", f"{totals['kinetic_setup_time_min']:.2f}")
    k3.metric("Kinetic total (min)", f"{totals['kinetic_total_time_min']:.2f}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Saw run (min)", f"{totals['grand_total_structural_cutting_time']:.2f}")
    s2.metric("Saw setup (min)", f"{totals['saw_setup_time_min']:.2f}")
    s3.metric("Saw total (min)", f"{totals['saw_total_time_min']:.2f}")

    # Existing drilling metric (kept separate)
    d1, d2 = st.columns(2)
    d1.metric("Plate drilling (min)", f"{totals['grand_total_plate_drilling_time']:.2f}")
    d2.metric("Plate bending (min)", f"{totals['grand_total_plate_bend_time']:.2f}")

    with st.expander("Estimate line items", expanded=False):
        st.dataframe(rows, use_container_width=True)

    # ------------------------------------------------------------
    # Plate nesting optimization (sheet cutting)
    # ------------------------------------------------------------
    with st.expander("Plate nesting optimization (sheet cutting)", expanded=False):
        st.caption(
            "This is a fast, estimator-friendly nesting heuristic. "
            "It supports **edge margin** (keep parts away from sheet edge) and **part border** (gap between parts). "
            "Parts are tried at **0°/90°** (optional).\n\n"
            "Note: DXF parts currently nest by **bounding box**. True profile nesting (DXF polygon packing) can be added next."
        )

        # Build required plate rectangles from the estimate
        plate_rows = [r for r in rows if str(r.get("Estimation Type", "")) == "Plate"]
        if not plate_rows:
            st.info("No plate parts in the estimate yet.")
        else:
            # Grouping helper: do not mix material/thickness in one nest
            groups = {}
            for r in plate_rows:
                mat = str(r.get("Material", "") or "").strip()
                try:
                    thk = float(r.get("Thickness (in)", 0.0) or 0.0)
                except Exception:
                    thk = 0.0
                thk_key = round(float(thk), 4)
                key = (mat, thk_key)
                groups.setdefault(key, []).append(r)

            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
            with c1:
                edge_margin = st.number_input("Edge margin (in)", min_value=0.0, value=0.5, step=0.125, key="plate_nest_edge")
            with c2:
                part_gap = st.number_input("Part border / gap (in)", min_value=0.0, value=0.25, step=0.125, key="plate_nest_gap")
            with c3:
                allow_rot = st.checkbox("Allow 90° rotation", value=True, key="plate_nest_rot")
            with c4:
                objective = st.selectbox("Objective", ["min_sheets", "max_utilization"], index=0, key="plate_nest_obj")

            st.markdown("#### Stock sheet sizes")
            st.caption("Add one or more sheet sizes. Qty is optional (blank = unlimited).")

            default_stock = pd.DataFrame(
                [
                    {"Width (in)": 96.0, "Height (in)": 240.0, "Qty (optional)": ""},
                    {"Width (in)": 120.0, "Height (in)": 240.0, "Qty (optional)": ""},
                ]
            )
            stock_df = st.data_editor(
                st.session_state["plate_stock_df"],
                use_container_width=True,
                num_rows="dynamic",
                key="plate_stock_editor",
                column_config={
                    "Width (in)": st.column_config.NumberColumn("Width (in)", min_value=1.0, step=1.0, required=False),
                    "Height (in)": st.column_config.NumberColumn("Height (in)", min_value=1.0, step=1.0, required=False),
                    "Qty (optional)": st.column_config.TextColumn("Qty (optional)", help="Leave blank for unlimited"),
                },
            )
            st.session_state["plate_stock_df"] = stock_df

            stock_sizes = []
            try:
                for _, rr in stock_df.iterrows():
                    w = float(rr.get("Width (in)") or 0.0)
                    h = float(rr.get("Height (in)") or 0.0)
                    qty_raw = rr.get("Qty (optional)")
                    qty = None
                    if qty_raw not in (None, ""):
                        try:
                            qty = int(str(qty_raw).strip())
                        except Exception:
                            qty = None
                    if w > 0 and h > 0:
                        stock_sizes.append({"width": w, "height": h, "qty": qty})
            except Exception:
                stock_sizes = []

            run_plate_nest = st.button("Run plate nesting optimization", type="primary", key="run_plate_nesting")
            if run_plate_nest:
                results_by_group = {}
                for (mat, thk), g_rows in groups.items():
                    required_parts = []
                    for r in g_rows:
                        try:
                            required_parts.append(
                                {
                                    "part_name": str(r.get("Part Name", "Plate")),
                                     "width": float(r.get("Width (in)", 0.0) or 0.0),
                                     "height": float(r.get("Length (in)", 0.0) or 0.0),
                                    "quantity": int(float(r.get("Quantity", 1) or 1)),
                                    "source": str(r.get("DXF Source", "")) or str(r.get("STEP Source File", "")) or "",
                                }
                            )
                        except Exception:
                            continue

                    sol = logic.optimize_plate_nesting(
                        required_parts,
                        stock_sizes,
                        edge_margin=float(edge_margin),
                        part_gap=float(part_gap),
                        allow_rot=bool(allow_rot),
                        objective=str(objective),
                    )
                    results_by_group[f"{mat} | {thk:.3f} in"] = sol

                st.session_state["plate_yield_results"] = results_by_group
                st.success("Plate nesting completed.")

            # Display last results (if any)
            results_by_group = st.session_state.get("plate_yield_results", {}) or {}
            if results_by_group:
                for group_label, sol in results_by_group.items():
                    st.markdown(f"### {group_label}")
                    best = (sol or {}).get("best")
                    if not best:
                        st.warning("No feasible nesting solution found for this group (check sheet sizes/margins).")
                        continue

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sheets used", str(best.get("sheets_used", 0)))
                    m2.metric("Utilization", f"{float(best.get('utilization', 0.0)) * 100.0:.1f}%")
                    m3.metric("Waste area (in²)", f"{float(best.get('total_waste_area', 0.0)):.0f}")
                    m4.metric("Heuristic", str(best.get("sort_key", "")))

                    unplaced = best.get("unplaced", []) or best.get("unplaced_parts", [])
                    if unplaced:
                        st.warning(f"Unplaced parts remaining: {len(unplaced)} (increase sheet size or reduce margins/gap)")

                    # Render each sheet
                    sheets = best.get("sheets", []) or []
                    if sheets:
                        cols = st.columns(2)
                        for idx, sh in enumerate(sheets):
                            img = _create_yield_image(sh, scale=4)
                            with cols[idx % 2]:
                                st.image(img, caption=f"Sheet {idx+1}: {sh['width']:.0f} x {sh['height']:.0f} in", use_container_width=True)

    csv_bytes = _export_csv_bytes(rows)
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
        page = st.radio("Go to", ["Plate", "Cone Calculator", "Structural", "Welding", "Summary"], index=0, key="nav_page")
        st.write("—")
        st.write(f"Items in estimate: **{len(st.session_state['estimate_parts'])}**")
        if st.button("Clear estimate", type="secondary", key="clear_estimate_btn"):
            _clear_estimate()
            st.rerun()
        if st.button("Sign out", type="secondary", key="sign_out_btn"):
            st.session_state["authenticated"] = False
            st.rerun()

    if page == "Plate":
        page_plate()
    elif page == "Cone Calculator":
        page_cone_calculator()
    elif page == "Structural":
        page_structural()
    elif page == "Welding":
        page_welding()
    else:
        page_summary()


if __name__ == "__main__":
    main()