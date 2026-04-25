"""Visualize one-step valid voxel states using project-native motion logic.

This file is standalone and does not modify old files.

Updates:
- Base heading is fixed to +X.
- All inputs are flattened into one global 1D step-id array.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from control.dstar_surface_3d import DStarLiteSurface3D, Node, OrientedNode, NORM, FACES, EDGE_DIRS

BASE_HEADING_FIXED = "+X"
START_CENTER_FIXED = (0.5, 0.5, 1.5)
START_FREE_VOXEL_FIXED = (0, 0, 1)
START_FACE_DIR_FIXED = "-Z"
FIXED_PLATFORM_FIXED = "base_platform"


@dataclass(frozen=True)
class StepCandidate:
    step_id: int  # global 1D id
    from_state: OrientedNode
    to_state: OrientedNode
    move_type: str
    start_free_voxel: Tuple[int, int, int]
    end_free_voxel: Tuple[int, int, int]
    start_surface_voxel: Tuple[int, int, int]
    end_surface_voxel: Tuple[int, int, int]


def _v_add(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _center_to_idx(p: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (int(round(p[0] - 0.5)), int(round(p[1] - 0.5)), int(round(p[2] - 0.5)))


def _node_center_from_free_voxel(v: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return (v[0] + 0.5, v[1] + 0.5, v[2] + 0.5)


def _vec_to_str(v: Tuple[float, float, float], ndigits: int = 2) -> str:
    return f"({round(v[0], ndigits)}, {round(v[1], ndigits)}, {round(v[2], ndigits)})"


def _classify_move_type(planner: DStarLiteSurface3D, cur: OrientedNode, nxt: OrientedNode) -> str:
    n0 = NORM[cur.node.face_dir]
    n1 = NORM[nxt.node.face_dir]
    dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]
    if dot == 0:
        return "edge_flip"

    relation = planner._classify_relative_parallel_move(cur, nxt.node)  # noqa: SLF001
    if relation is not None:
        return f"parallel_{relation}"
    return "parallel_relaxed"


def _heading_relation_to_base(heading_dir: str) -> str:
    """Describe heading relative to fixed base forward (+X)."""
    base_forward = NORM[BASE_HEADING_FIXED]
    base_left = (0, 1, 0)
    h = NORM[heading_dir]
    dot_f = h[0] * base_forward[0] + h[1] * base_forward[1] + h[2] * base_forward[2]
    dot_l = h[0] * base_left[0] + h[1] * base_left[1] + h[2] * base_left[2]
    if dot_f > 0:
        return "front"
    if dot_f < 0:
        return "back"
    if dot_l > 0:
        return "left"
    if dot_l < 0:
        return "right"
    if h[2] > 0:
        return "up"
    if h[2] < 0:
        return "down"
    return "other"


def _move_type_priority(move_type: str) -> int:
    order = {
        "parallel_front": 0,
        "parallel_front_left": 1,
        "parallel_front_right": 2,
        "parallel_left": 3,
        "parallel_right": 4,
        "parallel_relaxed": 5,
        "edge_flip": 6,
    }
    return order.get(move_type, 99)


def _heading_priority_for_id(heading_dir: str) -> int:
    # Requested ordering example for +X forward: back(0), left(1), right(2), then others.
    rel = _heading_relation_to_base(heading_dir)
    order = {
        "back": 0,
        "left": 1,
        "right": 2,
        "front": 3,
        "up": 4,
        "down": 5,
        "other": 6,
    }
    return order.get(rel, 99)


def _in_bounds(size_xyz: Tuple[int, int, int], v: Tuple[int, int, int]) -> bool:
    x, y, z = v
    return 0 <= x < size_xyz[0] and 0 <= y < size_xyz[1] and 0 <= z < size_xyz[2]


def _is_parallel_face_step_geom(a: Node, b: Node) -> bool:
    if a == b or a.face_dir != b.face_dir:
        return False

    n1 = NORM[a.face_dir]
    p1 = (a.pos[0] + 0.5 * n1[0], a.pos[1] + 0.5 * n1[1], a.pos[2] + 0.5 * n1[2])
    p2 = (b.pos[0] + 0.5 * n1[0], b.pos[1] + 0.5 * n1[1], b.pos[2] + 0.5 * n1[2])
    d = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

    dn = d[0] * n1[0] + d[1] * n1[1] + d[2] * n1[2]
    if abs(abs(dn) - 1.0) > 1e-9:
        return False

    tang = (d[0] - dn * n1[0], d[1] - dn * n1[1], d[2] - dn * n1[2])
    proj_dist = float(np.sqrt(tang[0] * tang[0] + tang[1] * tang[1] + tang[2] * tang[2]))
    return 0.0 < proj_dist < 2.0


def _build_logic_planner(size_xyz: Tuple[int, int, int], start_node: Node) -> DStarLiteSurface3D:
    occ = np.zeros(size_xyz, dtype=np.int8)
    start_free = _center_to_idx(start_node.pos)
    start_surface = _v_add(start_free, NORM[start_node.face_dir])
    if not _in_bounds(size_xyz, start_surface):
        raise ValueError("Start surface voxel is out of bounds for the current grid size")
    occ[start_surface[0], start_surface[1], start_surface[2]] = 1
    return _build_planner_quietly(occ, size_xyz, start_node, FIXED_PLATFORM_FIXED)


def _enumerate_successor_nodes_exhaustive(
    size_xyz: Tuple[int, int, int],
    start_node: Node,
    bound_by_grid: bool,
) -> List[Node]:
    start_free = _center_to_idx(start_node.pos)
    d_face = start_node.face_dir
    out: List[Node] = []

    # 1) Slide candidates (same face)
    for step_d in EDGE_DIRS[d_face]:
        u2 = _v_add(start_free, step_d)
        support = _v_add(u2, NORM[d_face])
        if (not bound_by_grid) or (_in_bounds(size_xyz, u2) and _in_bounds(size_xyz, support)):
            out.append(Node(_node_center_from_free_voxel(u2), d_face))

    # 2) Edge-flip candidates (orthogonal faces around same obstacle voxel)
    obs_v = _v_add(start_free, NORM[d_face])
    if (not bound_by_grid) or _in_bounds(size_xyz, obs_v):
        n1 = NORM[d_face]
        for f2 in FACES:
            if f2 == d_face:
                continue
            n2 = NORM[f2]
            if n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] != 0:
                continue
            u2 = (obs_v[0] - n2[0], obs_v[1] - n2[1], obs_v[2] - n2[2])
            if (not bound_by_grid) or _in_bounds(size_xyz, u2):
                out.append(Node(_node_center_from_free_voxel(u2), f2))

    # 3) Relaxed parallel candidates in local window (same face)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            for dz in (-2, -1, 0, 1, 2):
                v2 = (start_free[0] + dx, start_free[1] + dy, start_free[2] + dz)
                support = _v_add(v2, NORM[d_face])
                if bound_by_grid and (not _in_bounds(size_xyz, v2) or not _in_bounds(size_xyz, support)):
                    continue
                cand = Node(_node_center_from_free_voxel(v2), d_face)
                if _is_parallel_face_step_geom(start_node, cand):
                    out.append(cand)

    dedup: List[Node] = []
    seen = set()
    for node in out:
        key = (_center_to_idx(node.pos), node.face_dir)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(node)
    return dedup


def _build_occ_for_candidate(cand: StepCandidate, size_xyz: Tuple[int, int, int]) -> np.ndarray:
    occ = np.zeros(size_xyz, dtype=np.int8)
    for v in (cand.start_surface_voxel, cand.end_surface_voxel):
        if _in_bounds(size_xyz, v):
            occ[v[0], v[1], v[2]] = 1
    return occ


def _build_robot_occupied_voxels(cand: StepCandidate) -> set[Tuple[int, int, int]]:
    """Build robot-occupied voxels with special stacking/bridge rules."""
    start = cand.start_free_voxel
    end = cand.end_free_voxel
    occupied: set[Tuple[int, int, int]] = {start, end}

    def above(v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (v[0], v[1], v[2] + 1)

    # Layer relation rules between start and end free voxels.
    dz = start[2] - end[2]
    if dz == 0:
        occupied.add(above(start))
        occupied.add(above(end))
    elif dz == 1:
        occupied.add(above(end))
    elif dz == -1:
        occupied.add(above(start))

    # For edge flip, add the voxel exactly between start/end when it exists on grid coordinates.
    if cand.move_type == "edge_flip":
        sx, sy, sz = start
        ex, ey, ez = end
        if (sx + ex) % 2 == 0 and (sy + ey) % 2 == 0 and (sz + ez) % 2 == 0:
            mid = ((sx + ex) // 2, (sy + ey) // 2, (sz + ez) // 2)
            if mid != start and mid != end:
                occupied.add(mid)

    return occupied


def create_demo_occupancy(size_xyz: Tuple[int, int, int] = (8, 8, 6)) -> np.ndarray:
    """Create a demo world with floor + blocks to expose multiple step types."""
    X, Y, Z = size_xyz
    occ = np.zeros((X, Y, Z), dtype=np.int8)

    occ[:, :, 0] = 1  # floor

    extra_blocks = [
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
        (3, 2, 1),
    ]
    for x, y, z in extra_blocks:
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
            occ[x, y, z] = 1

    return occ


def _build_planner_quietly(
    occ: np.ndarray,
    size_xyz: Tuple[int, int, int],
    start_node: Node,
    fixed_platform: str,
) -> DStarLiteSurface3D:
    with contextlib.redirect_stdout(io.StringIO()):
        planner = DStarLiteSurface3D(
            occ,
            size_xyz,
            start_node,
            start_node,
            start_heading_dir=BASE_HEADING_FIXED,
            start_fixed_platform=fixed_platform,
        )
    return planner


def enumerate_global_flat_id_candidates(
    size_xyz: Tuple[int, int, int] = (8, 8, 6),
    occ: np.ndarray | None = None,
) -> tuple[np.ndarray, List[StepCandidate]]:
    """Flatten all valid one-step possibilities into one global 1D id array.

    The start center is fixed at START_CENTER_FIXED = (0.5, 0.5, 1.5).
    """
    exhaustive_logic_mode = occ is None
    if occ is None:
        occ = np.zeros(size_xyz, dtype=np.int8)
    else:
        size_xyz = tuple(int(v) for v in occ.shape)  # type: ignore[assignment]

    all_candidates: List[StepCandidate] = []

    sx, sy, sz = START_FREE_VOXEL_FIXED
    X, Y, Z = size_xyz
    if not (0 <= sx < X and 0 <= sy < Y and 0 <= sz < Z):
        raise ValueError(
            f"Fixed start center {START_CENTER_FIXED} maps to voxel {START_FREE_VOXEL_FIXED}, which is out of bounds for size {size_xyz}."
        )
    if not exhaustive_logic_mode and int(occ[sx, sy, sz]) != 0:
        raise ValueError(
            f"Fixed start center {START_CENTER_FIXED} maps to occupied voxel {START_FREE_VOXEL_FIXED}."
        )

    start_center = START_CENTER_FIXED
    start_node = Node(start_center, START_FACE_DIR_FIXED)
    try:
        planner = _build_logic_planner(size_xyz, start_node) if exhaustive_logic_mode else _build_planner_quietly(occ, size_xyz, start_node, FIXED_PLATFORM_FIXED)
        start_state = OrientedNode(start_node, BASE_HEADING_FIXED, FIXED_PLATFORM_FIXED)
        if not planner.valid_state(start_state):
            return occ, []
    except Exception:
        return occ, []

    if exhaustive_logic_mode:
        successor_nodes = _enumerate_successor_nodes_exhaustive(size_xyz, start_node, bound_by_grid=False)
        successors: List[OrientedNode] = []
        for next_node in successor_nodes:
            for heading_dir in planner._allowed_next_headings(start_state, next_node):  # noqa: SLF001
                successors.append(OrientedNode(next_node, heading_dir, "end_platform"))
    else:
        successors = planner.successors(start_state)

    start_free = _center_to_idx(start_state.node.pos)
    start_surface = _v_add(start_free, NORM[start_state.node.face_dir])
    for nxt in successors:
        end_free = _center_to_idx(nxt.node.pos)
        end_surface = _v_add(end_free, NORM[nxt.node.face_dir])
        all_candidates.append(
            StepCandidate(
                step_id=-1,
                from_state=start_state,
                to_state=nxt,
                move_type=_classify_move_type(planner, start_state, nxt),
                start_free_voxel=start_free,
                end_free_voxel=end_free,
                start_surface_voxel=start_surface,
                end_surface_voxel=end_surface,
            )
        )

    all_candidates.sort(
        key=lambda c: (
            _move_type_priority(c.move_type),
            c.end_free_voxel,
            _heading_priority_for_id(c.to_state.heading_dir),
            c.end_free_voxel,
            c.to_state.node.face_dir,
            c.to_state.heading_dir,
            c.to_state.fixed_platform,
        )
    )

    normalized: List[StepCandidate] = []
    for idx, c in enumerate(all_candidates):
        normalized.append(
            StepCandidate(
                step_id=idx,
                from_state=c.from_state,
                to_state=c.to_state,
                move_type=c.move_type,
                start_free_voxel=c.start_free_voxel,
                end_free_voxel=c.end_free_voxel,
                start_surface_voxel=c.start_surface_voxel,
                end_surface_voxel=c.end_surface_voxel,
            )
        )

    return occ, normalized


def describe_step_id_table(candidates: List[StepCandidate]) -> str:
    lines = [
        "Global Step ID mapping (flattened 1D): "
        f"base heading={BASE_HEADING_FIXED}, start center={START_CENTER_FIXED}, start face={START_FACE_DIR_FIXED}, fixed={FIXED_PLATFORM_FIXED}"
    ]
    if not candidates:
        lines.append("  <no valid one-step candidates>")
        return "\n".join(lines)

    for c in candidates:
        f = c.to_state
        end_idx = _center_to_idx(f.node.pos)
        dz = end_idx[2] - c.start_free_voxel[2]
        end_heading_rel = _heading_relation_to_base(f.heading_dir)
        lines.append(
            "  "
            f"{c.step_id:3d}: type={c.move_type:16s} "
            f"from=({c.start_free_voxel}, {c.from_state.node.face_dir}, {c.from_state.fixed_platform}) "
            f"to=({end_idx}, {f.node.face_dir}, heading={f.heading_dir}/{end_heading_rel}, fixed={f.fixed_platform}) dz={dz:+d}"
        )
    return "\n".join(lines)


def plot_valid_step_voxel(
    step_id: int,
    size_xyz: Tuple[int, int, int] = (8, 8, 6),
    occ: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Plot one selected global-step candidate with requested red/green voxel semantics."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    occ_grid, candidates = enumerate_global_flat_id_candidates(size_xyz=size_xyz, occ=occ)

    if not candidates:
        raise ValueError("No valid one-step candidates in current occupancy.")
    if step_id < 0 or step_id >= len(candidates):
        raise ValueError(f"step_id out of range: {step_id}, valid range [0, {len(candidates)-1}]")

    cand = candidates[step_id]
    start_state = cand.from_state

    if occ is None:
        occ_grid = _build_occ_for_candidate(cand, size_xyz)

    planner = _build_planner_quietly(
        occ_grid,
        tuple(int(v) for v in occ_grid.shape),
        start_state.node,
        start_state.fixed_platform,
    )

    red_voxels = _build_robot_occupied_voxels(cand)
    green_voxels = {cand.start_surface_voxel, cand.end_surface_voxel}

    all_voxels = sorted(red_voxels | green_voxels)
    xs = [v[0] for v in all_voxels]
    ys = [v[1] for v in all_voxels]
    zs = [v[2] for v in all_voxels]
    min_x, max_x = min(xs) - 1, max(xs) + 2
    min_y, max_y = min(ys) - 1, max(ys) + 2
    min_z, max_z = min(zs) - 1, max(zs) + 2

    shape = (max_x - min_x, max_y - min_y, max_z - min_z)
    filled = np.zeros(shape, dtype=bool)
    colors = np.zeros(shape + (4,), dtype=float)

    def paint(v: Tuple[int, int, int], rgba: Tuple[float, float, float, float]) -> None:
        ix = v[0] - min_x
        iy = v[1] - min_y
        iz = v[2] - min_z
        if ix < 0 or iy < 0 or iz < 0 or ix >= shape[0] or iy >= shape[1] or iz >= shape[2]:
            return
        filled[ix, iy, iz] = True
        colors[ix, iy, iz] = rgba

    for v in red_voxels:
        paint(v, (1.0, 0.0, 0.0, 0.35))
    for v in green_voxels:
        paint(v, (0.0, 1.0, 0.0, 0.35))

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    gx, gy, gz = np.indices(np.array(shape) + 1)
    ax.voxels(
        gx + min_x,
        gy + min_y,
        gz + min_z,
        filled,
        facecolors=colors,
        edgecolor="k",
        linewidth=0.8,
    )

    occ_xyz = np.argwhere(occ_grid == 1)
    if len(occ_xyz) > 0:
        ax.scatter(
            occ_xyz[:, 0] + 0.5,
            occ_xyz[:, 1] + 0.5,
            occ_xyz[:, 2] + 0.5,
            s=4,
            c="gray",
            alpha=0.15,
        )

    start_mid = planner.node_to_face_midpoint(start_state.node)
    end_mid = planner.node_to_face_midpoint(cand.to_state.node)
    arrow_len = 0.32

    for tag, state, mid, center in [
        ("start", start_state, start_mid, start_state.node.pos),
        ("end", cand.to_state, end_mid, cand.to_state.node.pos),
    ]:
        hx, hy, hz = NORM[state.heading_dir]
        ax.quiver(
            [mid[0]],
            [mid[1]],
            [mid[2]],
            [hx * arrow_len],
            [hy * arrow_len],
            [hz * arrow_len],
            color="tab:orange",
            linewidth=1.8,
            arrow_length_ratio=0.35,
        )
        ax.scatter([mid[0]], [mid[1]], [mid[2]], s=70, marker="o", color="tab:blue")
        ax.text(
            mid[0],
            mid[1],
            mid[2] + 0.08,
            f"{tag}: heading={state.heading_dir}/{_heading_relation_to_base(state.heading_dir)}, center={_vec_to_str(center)}, face={state.node.face_dir}, fixed={state.fixed_platform}",
            color="tab:orange",
            fontsize=8,
        )

    show_title = title or (
        f"Step ID {step_id} | {cand.move_type} | from face={start_state.node.face_dir} "
        f"to face={cand.to_state.node.face_dir} | base_heading={BASE_HEADING_FIXED}"
    )
    ax.set_title(show_title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=(1.0, 0.0, 0.0, 0.35), edgecolor="k", label="robot occupied voxels"),
        Patch(facecolor=(0.0, 1.0, 0.0, 0.35), edgecolor="k", label="start/end surface voxels"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.show()


def print_step_table(size_xyz: Tuple[int, int, int] = (8, 8, 6)) -> None:
    _, candidates = enumerate_global_flat_id_candidates(size_xyz=size_xyz)
    print(describe_step_id_table(candidates))
    print(f"\nTotal one-step candidates: {len(candidates)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize one-step valid voxel state by flattened global step id"
    )
    parser.add_argument("step_id", type=int, nargs="?", default=0, help="global step id in flattened 1D id array")
    parser.add_argument("--list", action="store_true", help="print flattened global id mapping and exit")

    args = parser.parse_args()

    if args.list:
        print_step_table()
    else:
        plot_valid_step_voxel(step_id=args.step_id)
