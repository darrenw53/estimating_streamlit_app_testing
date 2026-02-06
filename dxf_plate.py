"""DXF utilities for Plate batch import.

Plate-only helper that extracts part sizes and cutting metrics from DXF.

What we compute (per detected part):
  - Bounding box width/length (axis-aligned) in inches
  - True cut perimeter in inches: outer profile + sum(hole perimeters)
  - Hole count
  - Total hole circumference

Nested DXFs (multiple parts in one file) are supported by identifying
multiple outer profiles (closed loops not contained inside any other loop).

We ignore common etch/scribe/text/dimension layers (configurable).

Dependencies: ezdxf, shapely
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional, Tuple

import tempfile

import ezdxf
from ezdxf.path import make_path

from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize, snap
from shapely.prepared import prep

from PIL import Image, ImageDraw
import base64
import io


DEFAULT_IGNORE_LAYER_SUBSTRINGS = [
    "ETCH",
    "SCRIBE",
    "ENGRAVE",
    "MARK",
    "TEXT",
    "DIM",
    "CENTER",
    "CL",
]


@dataclass(frozen=True)
class DetectedPart:
    part_name: str
    bbox_w_in: float
    bbox_l_in: float
    cut_perimeter_in: float
    hole_count: int
    hole_circumference_in: float


@dataclass(frozen=True)
class DetectedPartGeometry:
    """Detected part plus geometry (scaled to inches) for preview rendering."""

    part: DetectedPart
    outer_poly_in: Polygon
    holes_poly_in: List[Polygon]


def _scale_factor(units: str) -> float:
    u = (units or "in").strip().lower()
    if u in {"mm", "millimeter", "millimeters"}:
        return 1.0 / 25.4
    # Default: inches
    return 1.0


def _contains_with_tol(container: Polygon, item: Polygon, tol: float) -> bool:
    """Robust containment with tolerance.

    After flattening curves, boundaries can end up microscopically outside/inside
    due to numerical effects. Buffering the container a bit makes `contains`
    behave as users expect.
    """
    try:
        if tol and tol > 0:
            return container.buffer(tol).contains(item)
        return container.contains(item)
    except Exception:
        return False


def _layer_is_ignored(layer_name: str, ignore_substrings: Iterable[str]) -> bool:
    name = (layer_name or "").upper()
    for s in ignore_substrings:
        if s and s.upper() in name:
            return True
    return False


def _polygon_from_entity(entity, flatten_tol: float) -> Optional[Polygon]:
    """Try to convert a DXF entity into a closed shapely Polygon.

    We only return polygons for closed geometry. Open paths return None.
    Splines/ellipses are flattened based on flatten_tol.
    """
    try:
        path = make_path(entity)
    except Exception:
        return None

    if path is None:
        return None

    if not getattr(path, "is_closed", False):
        return None

    # Flatten to vertices
    try:
        vertices = list(path.flattening(distance=flatten_tol))
    except Exception:
        return None

    if len(vertices) < 4:
        return None

    coords = [(float(v.x), float(v.y)) for v in vertices]
    # Ensure closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        poly = Polygon(coords)
    except Exception:
        return None

    if not poly.is_valid or poly.area <= 0:
        return None

    return poly


def _collect_closed_polygons(doc, ignore_substrings: Iterable[str], flatten_tol: float) -> List[Polygon]:
    """Collect polygons from DXF geometry.

    We prefer a robust approach that works for DXFs where the part outline is
    represented as many LINE/ARC segments (i.e., not a single closed polyline).

    Strategy:
      1) Convert eligible entities to ezdxf Paths
      2) Flatten paths into line segments (LineStrings)
      3) Use shapely.polygonize to recover closed loops

    This greatly reduces "false parts" where only holes are closed entities.
    """
    msp = doc.modelspace()
    segments: List[LineString] = []

    SKIP_TYPES = {"TEXT", "MTEXT", "DIMENSION", "LEADER", "MLEADER", "HATCH"}

    def handle_entity(ent) -> None:
        t = ent.dxftype()
        if t in SKIP_TYPES:
            return
        layer = getattr(ent.dxf, "layer", "")
        if _layer_is_ignored(layer, ignore_substrings):
            return
        try:
            path = make_path(ent)
        except Exception:
            return
        if path is None:
            return
        try:
            verts = list(path.flattening(distance=flatten_tol))
        except Exception:
            return
        if len(verts) < 2:
            return
        coords = [(float(v.x), float(v.y)) for v in verts]
        try:
            segments.append(LineString(coords))
        except Exception:
            return

    for e in msp:
        if e.dxftype() == "INSERT":
            # Skip common SolidWorks/annotation blocks (notes, center marks, etc.)
            try:
                bname = str(getattr(e.dxf, "name", "")).upper()
            except Exception:
                bname = ""
            if bname.startswith("*") or any(s in bname for s in [
                "SW_NOTE",
                "SW_CENTERMARK",
                "CENTERMARK",
                "CENTER_MARK",
                "DIM",
                "NOTE",
            ]):
                continue
            # Explode block references into their transformed virtual entities.
            # This is common for DXFs exported from CAD where geometry is in blocks.
            try:
                for ve in e.virtual_entities():
                    handle_entity(ve)
            except Exception:
                continue
        else:
            handle_entity(e)

    if not segments:
        return []

    try:
        merged = unary_union(segments)
        # Attempt to close tiny gaps so polygonize can recover outer profiles.
        join_tol = max(float(flatten_tol) * 5.0, 1e-4)
        try:
            merged = snap(merged, merged, join_tol)
        except Exception:
            pass
        polys = list(polygonize(merged))
    except Exception:
        polys = []

    # Filter invalid / zero-area
    out: List[Polygon] = []
    for p in polys:
        try:
            if p.is_valid and p.area > 0:
                out.append(p)
        except Exception:
            continue
    return out


def _assign_outers_and_holes(polys: List[Polygon], tol: float) -> List[Tuple[Polygon, List[Polygon]]]:
    """Return list of (outer_poly, holes[]) for each detected part.

    This uses an "enclosing count" approach (how many other polygons contain a
    polygon) with a small tolerance buffer to avoid false "not contained" cases.
    """
    if not polys:
        return []

    # Sort by area descending for a stable ordering and for boundary heuristics
    polys_sorted = sorted(polys, key=lambda p: p.area, reverse=True)

    # Heuristic: if the largest polygon looks like a sheet/border boundary,
    # drop it (common in some exports).
    if len(polys_sorted) >= 3:
        p0 = polys_sorted[0]
        p1 = polys_sorted[1]
        if p0.area > 4.0 * p1.area:
            contains_n = sum(1 for p in polys_sorted[1:] if _contains_with_tol(p0, p, tol))
            if contains_n >= int(0.6 * (len(polys_sorted) - 1)):
                polys_sorted = polys_sorted[1:]

    # Precompute containment counts
    containment_count: List[int] = [0] * len(polys_sorted)
    for i, p in enumerate(polys_sorted):
        for j, q in enumerate(polys_sorted):
            if i == j:
                continue
            # Only larger areas can contain smaller ones (fast prune)
            if q.area <= p.area:
                continue
            if _contains_with_tol(q, p, tol):
                containment_count[i] += 1

    # Outers are those not contained by any other polygon
    outers: List[Polygon] = [p for p, c in zip(polys_sorted, containment_count) if c == 0]

    # Assign holes to the smallest outer that contains them
    holes_by_outer: Dict[int, List[Polygon]] = {i: [] for i in range(len(outers))}
    for p in polys_sorted:
        # Skip if it's an outer
        if any(p.equals(o) for o in outers):
            continue
        containing: List[Tuple[int, float]] = []
        for i, o in enumerate(outers):
            if _contains_with_tol(o, p, tol):
                containing.append((i, o.area))
        if containing:
            containing.sort(key=lambda x: x[1])
            holes_by_outer[containing[0][0]].append(p)

    return [(outers[i], holes_by_outer[i]) for i in range(len(outers))]


def parse_dxf_plate_parts(
    file_bytes: bytes,
    filename: str = "part.dxf",
    units: str = "in",
    scale: float = 1.0,
    ignore_layer_substrings: Optional[List[str]] = None,
    flatten_tol: float = 0.01,
) -> List[DetectedPart]:
    """Parse DXF (bytes) and return detected plate parts.

    Parameters
    ----------
    units:
        "in" or "mm". Output is always inches.
    ignore_layer_substrings:
        List of case-insensitive substrings; entities on layers containing
        any of these are ignored.
    flatten_tol:
        Flattening tolerance used when approximating curves.
    """
    ignore = ignore_layer_substrings or list(DEFAULT_IGNORE_LAYER_SUBSTRINGS)
    # Total scale to inches: units scale * user-provided scale multiplier
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    sf = _scale_factor(units) * (scale if scale > 0 else 1.0)

    # ezdxf is most reliable reading from a temporary file
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = ezdxf.readfile(tmp.name)

    polys = _collect_closed_polygons(doc, ignore_substrings=ignore, flatten_tol=flatten_tol)
    # Use a small tolerance for containment tests. Tie it to flatten_tol.
    tol = max(float(flatten_tol) * 2.0, 1e-6)
    groups = _assign_outers_and_holes(polys, tol=tol)

    parts: List[DetectedPart] = []
    base = filename.rsplit(".", 1)[0]

    for idx, (outer, holes) in enumerate(groups, start=1):
        # scale geometry to inches
        outer_s = outer
        holes_s = holes
        if sf != 1.0:
            # shapely scale via manual coordinate scaling
            def _scale_poly(p: Polygon) -> Polygon:
                x, y = p.exterior.coords.xy
                coords = [(xi * sf, yi * sf) for xi, yi in zip(x, y)]
                return Polygon(coords)

            outer_s = _scale_poly(outer)
            holes_s = [_scale_poly(h) for h in holes]

        minx, miny, maxx, maxy = outer_s.bounds
        bbox_w = float(maxx - minx)
        bbox_l = float(maxy - miny)

        outer_perim = float(outer_s.exterior.length)
        hole_perims = [float(h.exterior.length) for h in holes_s]
        hole_circ = float(sum(hole_perims))

        cut_perim = outer_perim + hole_circ
        hole_count = len(holes_s)

        part_name = f"{base} - Part {idx}" if len(groups) > 1 else base

        parts.append(
            DetectedPart(
                part_name=part_name,
                bbox_w_in=round(bbox_w, 3),
                bbox_l_in=round(bbox_l, 3),
                cut_perimeter_in=round(cut_perim, 3),
                hole_count=hole_count,
                hole_circumference_in=round(hole_circ, 3),
            )
        )

    return parts


def parse_dxf_plate_single_part(
    file_bytes: bytes,
    filename: str = "part.dxf",
    units: str = "in",
    scale: float = 1.0,
    ignore_layer_substrings: Optional[List[str]] = None,
    flatten_tol: float = 0.01,
    strict_single: bool = False,
) -> Optional[DetectedPart]:
    """Parse DXF (bytes) and return a *single* detected plate part.

    This is the safest import mode for estimators: one DXF => one plate item.

    Selection rules:
      - If no closed profile is detected, returns None.
      - If multiple outer profiles are detected:
          * strict_single=False: pick the largest outer profile by area.
          * strict_single=True : raise ValueError.

    Parameters
    ----------
    units:
        "in" or "mm". Output is always inches.
    scale:
        User-specified multiplier applied after unit conversion.
    strict_single:
        If True, fail when multiple outer profiles are found.
    """

    ignore = ignore_layer_substrings or list(DEFAULT_IGNORE_LAYER_SUBSTRINGS)
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    sf = _scale_factor(units) * (scale if scale > 0 else 1.0)

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = ezdxf.readfile(tmp.name)

    polys = _collect_closed_polygons(doc, ignore_substrings=ignore, flatten_tol=flatten_tol)
    tol = max(float(flatten_tol) * 2.0, 1e-6)
    groups = _assign_outers_and_holes(polys, tol=tol)

    if not groups:
        return None

    if strict_single and len(groups) > 1:
        base = filename.rsplit(".", 1)[0]
        raise ValueError(f"{base}: multiple outer profiles detected ({len(groups)}).")

    # Pick largest outer by area
    outer, holes = max(groups, key=lambda g: float(getattr(g[0], "area", 0.0)))

    outer_s = outer
    holes_s = holes
    if sf != 1.0:
        def _scale_poly(p: Polygon) -> Polygon:
            x, y = p.exterior.coords.xy
            coords = [(xi * sf, yi * sf) for xi, yi in zip(x, y)]
            return Polygon(coords)

        outer_s = _scale_poly(outer)
        holes_s = [_scale_poly(h) for h in holes]

    minx, miny, maxx, maxy = outer_s.bounds
    bbox_w = float(maxx - minx)
    bbox_l = float(maxy - miny)

    outer_perim = float(outer_s.exterior.length)
    hole_perims = [float(h.exterior.length) for h in holes_s]
    hole_circ = float(sum(hole_perims))
    cut_perim = outer_perim + hole_circ

    base = filename.rsplit(".", 1)[0]
    return DetectedPart(
        part_name=base,
        bbox_w_in=round(bbox_w, 3),
        bbox_l_in=round(bbox_l, 3),
        cut_perimeter_in=round(cut_perim, 3),
        hole_count=len(holes_s),
        hole_circumference_in=round(hole_circ, 3),
    )


def parse_dxf_plate_single_part_geometry(
    file_bytes: bytes,
    filename: str = "part.dxf",
    units: str = "in",
    scale: float = 1.0,
    ignore_layer_substrings: Optional[List[str]] = None,
    flatten_tol: float = 0.01,
    strict_single: bool = False,
) -> Optional[DetectedPartGeometry]:
    """Like parse_dxf_plate_single_part, but also returns geometry for rendering."""

    ignore = ignore_layer_substrings or list(DEFAULT_IGNORE_LAYER_SUBSTRINGS)
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    sf = _scale_factor(units) * (scale if scale > 0 else 1.0)

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        doc = ezdxf.readfile(tmp.name)

    polys = _collect_closed_polygons(doc, ignore_substrings=ignore, flatten_tol=flatten_tol)
    tol = max(float(flatten_tol) * 2.0, 1e-6)
    groups = _assign_outers_and_holes(polys, tol=tol)

    if not groups:
        return None

    if strict_single and len(groups) > 1:
        base = filename.rsplit(".", 1)[0]
        raise ValueError(f"{base}: multiple outer profiles detected ({len(groups)}).")

    outer, holes = max(groups, key=lambda g: float(getattr(g[0], "area", 0.0)))

    outer_s = outer
    holes_s = holes
    if sf != 1.0:
        def _scale_poly(p: Polygon) -> Polygon:
            x, y = p.exterior.coords.xy
            coords = [(xi * sf, yi * sf) for xi, yi in zip(x, y)]
            return Polygon(coords)

        outer_s = _scale_poly(outer)
        holes_s = [_scale_poly(h) for h in holes]

    minx, miny, maxx, maxy = outer_s.bounds
    bbox_w = float(maxx - minx)
    bbox_l = float(maxy - miny)
    outer_perim = float(outer_s.exterior.length)
    hole_perims = [float(h.exterior.length) for h in holes_s]
    hole_circ = float(sum(hole_perims))
    cut_perim = outer_perim + hole_circ

    base = filename.rsplit(".", 1)[0]
    part = DetectedPart(
        part_name=base,
        bbox_w_in=round(bbox_w, 3),
        bbox_l_in=round(bbox_l, 3),
        cut_perimeter_in=round(cut_perim, 3),
        hole_count=len(holes_s),
        hole_circumference_in=round(hole_circ, 3),
    )
    return DetectedPartGeometry(part=part, outer_poly_in=outer_s, holes_poly_in=list(holes_s))


def render_part_thumbnail_data_uri(
    geom: DetectedPartGeometry,
    size_px: int = 140,
    pad_px: int = 10,
    outer_width: int = 3,
    hole_width: int = 2,
) -> str:
    """Render a small PNG thumbnail for a part and return as a data URI."""

    outer = geom.outer_poly_in
    holes = geom.holes_poly_in

    minx, miny, maxx, maxy = outer.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        # Fallback: blank image
        img = Image.new("RGB", (size_px, size_px), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # Fit to canvas
    usable = max(1, size_px - 2 * pad_px)
    scale = min(usable / w, usable / h)

    def tx(x: float) -> float:
        return (x - minx) * scale + pad_px

    # Invert Y for image coordinates
    def ty(y: float) -> float:
        return (maxy - y) * scale + pad_px

    img = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(img)

    def draw_poly(p: Polygon, width: int) -> None:
        coords = list(p.exterior.coords)
        pts = [(tx(x), ty(y)) for x, y in coords]
        if len(pts) >= 2:
            draw.line(pts, fill="black", width=width, joint="curve")

    draw_poly(outer, outer_width)
    for hp in holes:
        draw_poly(hp, hole_width)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parts_to_rows(parts: List[DetectedPart]) -> List[Dict[str, Any]]:
    """Convenience for turning DetectedPart into simple dict rows."""
    rows: List[Dict[str, Any]] = []
    for p in parts:
        rows.append(
            {
                "Part Name": p.part_name,
                "Width (in)": p.bbox_w_in,
                "Length (in)": p.bbox_l_in,
                "True Cut Perimeter (in)": p.cut_perimeter_in,
                "Hole Count": p.hole_count,
                "Total Hole Circumference (in)": p.hole_circumference_in,
            }
        )
    return rows
