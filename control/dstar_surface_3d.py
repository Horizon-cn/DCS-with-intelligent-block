import heapq
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set

INF = 10**15

# --- Face encoding ---
FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
NORM = {
    "+X": (1, 0, 0), "-X": (-1, 0, 0),
    "+Y": (0, 1, 0), "-Y": (0, -1, 0),
    "+Z": (0, 0, 1), "-Z": (0, 0, -1),
}

OPPOSITE_FACE = {
    "+X": "-X", "-X": "+X",
    "+Y": "-Y", "-Y": "+Y",
    "+Z": "-Z", "-Z": "+Z",
}

# For each face, the 4 in-plane edge directions (orthogonal to the face normal)
EDGE_DIRS = {
    "+X": [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "-X": [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "+Y": [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
    "-Y": [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
    "+Z": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    "-Z": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
}

@dataclass(frozen=True)
class Node:
    """A planning node is a specific exposed face adjacent to a free voxel."""
    pos: Tuple[float, float, float]
    face_dir: str


class LazyPQ:
    """Priority queue with lazy deletion via key re-checking."""
    def __init__(self):
        self.h = []
        self.counter = 0

    def push(self, key, node):
        self.counter += 1
        heapq.heappush(self.h, (key[0], key[1], self.counter, node))

    def empty(self):
        return len(self.h) == 0

    def top(self):
        return self.h[0] if self.h else None

    def pop_valid(self, key_fn):
        while self.h:
            k1, k2, _, node = heapq.heappop(self.h)
            k_now = key_fn(node)
            if (k1, k2) == k_now:
                return (k1, k2), node
            # else outdated entry; drop
        return None, None

    def peek_key(self):
        # returns smallest key in heap (may be outdated, but ok for loop condition; safe but can be conservative)
        if not self.h:
            return (INF, INF)
        return (self.h[0][0], self.h[0][1])


class DStarLiteSurface3D:
    def __init__(self, occ, size_xyz: Tuple[int, int, int], start: Node, goal: Node):
        """
        occ: 3D occupancy container supporting occ[x,y,z] -> 0/1 (free/blocked)
             e.g. numpy array with shape (X,Y,Z) or dict-like with __getitem__
        size_xyz: (X,Y,Z)
        start/goal: Node((x,y,z), face) where each node is one valid exposed face.
        """
        self.occ = occ
        self.X, self.Y, self.Z = size_xyz
        self.visited_faces: Set[Node] = set()
        self.start = start
        self.goal = goal
        # Save original start/goal for visualization (since self.start gets modified during planning)
        self.start_orig = start
        self.goal_orig = goal

        if not self.valid_node(start):
            raise ValueError(f"start {start} is not a valid exposed obstacle face node.")
        if not self.valid_node(goal):
            raise ValueError(f"goal {goal} is not a valid exposed obstacle face node.")

        self.g: Dict[Node, int] = {}
        self.rhs: Dict[Node, int] = {}
        self.OPEN = LazyPQ()
        self.km = 0

        self.rhs[self.goal] = 0
        self.g[self.goal] = INF
        self.OPEN.push(self.key(self.goal), self.goal)
        start_mid = self.node_to_face_midpoint(self.start)
        goal_mid = self.node_to_face_midpoint(self.goal)
        print(
            "Initialized D* Lite with "
            f"start(center={self.start.pos}, face={self.start.face_dir}, face_mid={start_mid}) "
            f"and goal(center={self.goal.pos}, face={self.goal.face_dir}, face_mid={goal_mid})"
        )
        
        # --- batch update mechanism ---
        self.update_buffer: Dict[Tuple[int,int,int], int] = {}  # voxel -> new_occ value
        self.affected_nodes_cache: Set[Node] = set()  # accumulated affected nodes

    # --- basic voxel helpers ---
    def in_bounds(self, v: Tuple[int,int,int]) -> bool:
        x,y,z = v
        return 0 <= x < self.X and 0 <= y < self.Y and 0 <= z < self.Z

    def occ_at(self, v: Tuple[int,int,int]) -> int:
        x,y,z = v
        return int(self.occ[x,y,z])

    def addv(self, a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

    def idx_to_center(self, v: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return (v[0] + 0.5, v[1] + 0.5, v[2] + 0.5)

    def center_to_idx(self, p: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return (int(round(p[0] - 0.5)), int(round(p[1] - 0.5)), int(round(p[2] - 0.5)))

    def node_to_face_midpoint(self, node: Node) -> Tuple[float, float, float]:
        """Convert a face-node to the geometric midpoint of that face."""
        n = NORM[node.face_dir]
        return (
            node.pos[0] + 0.5 * n[0],
            node.pos[1] + 0.5 * n[1],
            node.pos[2] + 0.5 * n[2],
        )

    def _norm3(self, v: Tuple[float, float, float]) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    def _dot3(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _cross3(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _point_strictly_inside_obstacle(self, p: Tuple[float, float, float], eps: float = 1e-4) -> bool:
        """Return True only if point is strictly inside an occupied voxel interior (boundary contact allowed)."""
        ix = int(math.floor(p[0]))
        iy = int(math.floor(p[1]))
        iz = int(math.floor(p[2]))
        if not self.in_bounds((ix, iy, iz)):
            return True
        if self.occ_at((ix, iy, iz)) == 0:
            return False

        fx = p[0] - ix
        fy = p[1] - iy
        fz = p[2] - iz
        return (eps+1e-9 < fx < 1.0 - eps-1e-9) and (eps+1e-9 < fy < 1.0 - eps-1e-9) and (eps+1e-9 < fz < 1.0 - eps-1e-9)

    def _segment_is_clear(self, a: Tuple[float, float, float], b: Tuple[float, float, float], sample_dist: float = 0.1) -> bool:
        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        dist = self._norm3(d)
        if dist < 1e-9:
            return not self._point_strictly_inside_obstacle(a)

        n_samples = max(2, int(math.ceil(dist / max(1e-6, sample_dist))))
        for i in range(n_samples + 1):
            t = i / n_samples
            p = (a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t)
            if self._point_strictly_inside_obstacle(p):
                return False
        return True

    def _segment_in_shell_is_clear(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        center: Tuple[float, float, float],
        r_min: float,
        r_max: float,
        sample_dist: float = 0.1,
        shell_eps: float = 1e-3,
    ) -> bool:
        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        dist = self._norm3(d)
        if dist < 1e-9:
            ra = self._norm3((a[0] - center[0], a[1] - center[1], a[2] - center[2]))
            if ra < r_min - shell_eps or ra > r_max + shell_eps:
                return False
            return not self._point_strictly_inside_obstacle(a)

        n_samples = max(2, int(math.ceil(dist / max(1e-6, sample_dist))))
        for i in range(n_samples + 1):
            t = i / n_samples
            p = (a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t)
            rp = self._norm3((p[0] - center[0], p[1] - center[1], p[2] - center[2]))
            if rp < r_min - shell_eps or rp > r_max + shell_eps:
                return False
            if self._point_strictly_inside_obstacle(p):
                return False
        return True

    def _fibonacci_unit_dirs(self, n: int) -> List[Tuple[float, float, float]]:
        if n <= 0:
            return []
        if n == 1:
            return [(1.0, 0.0, 0.0)]

        dirs: List[Tuple[float, float, float]] = []
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n):
            y = 1.0 - (2.0 * i) / (n - 1)
            r = math.sqrt(max(0.0, 1.0 - y * y))
            th = golden_angle * i
            dirs.append((math.cos(th) * r, y, math.sin(th) * r))
        return dirs

    def _arc_points_around_center(
        self,
        center: Tuple[float, float, float],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        min_steps: int = 8,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """Sample the shorter circular arc from start to end with given center."""
        v1 = (start[0] - center[0], start[1] - center[1], start[2] - center[2])
        v2 = (end[0] - center[0], end[1] - center[1], end[2] - center[2])
        r1 = self._norm3(v1)
        r2 = self._norm3(v2)
        if r1 < 1e-9 or r2 < 1e-9:
            return None

        # A center-based circular trajectory requires approximately equal radii.
        if abs(r1 - r2) > 1e-3:
            return None

        inv_r = 1.0 / r1
        u1 = (v1[0] * inv_r, v1[1] * inv_r, v1[2] * inv_r)
        u2 = (v2[0] * inv_r, v2[1] * inv_r, v2[2] * inv_r)

        dot = max(-1.0, min(1.0, self._dot3(u1, u2)))
        angle = math.acos(dot)
        if angle < 1e-9:
            return [start, end]

        axis = self._cross3(u1, u2)
        axis_norm = self._norm3(axis)
        if axis_norm < 1e-9:
            return None
        axis = (axis[0] / axis_norm, axis[1] / axis_norm, axis[2] / axis_norm)

        steps = max(min_steps, int(math.ceil(angle / (math.pi / 18.0))))
        pts: List[Tuple[float, float, float]] = []
        for i in range(steps + 1):
            t = i / steps
            th = angle * t
            ct = math.cos(th)
            st = math.sin(th)
            kxu1 = self._cross3(axis, u1)
            kdu = self._dot3(axis, u1)
            # Rodrigues' rotation formula
            ur = (
                u1[0] * ct + kxu1[0] * st + axis[0] * kdu * (1.0 - ct),
                u1[1] * ct + kxu1[1] * st + axis[1] * kdu * (1.0 - ct),
                u1[2] * ct + kxu1[2] * st + axis[2] * kdu * (1.0 - ct),
            )
            pts.append((center[0] + ur[0] * r1, center[1] + ur[1] * r1, center[2] + ur[2] * r1))
        return pts

    def _arc_path_is_clear(
        self,
        center: Tuple[float, float, float],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> bool:
        arc_pts = self._arc_points_around_center(center, start, end)
        if arc_pts is None:
            return False
        for i in range(len(arc_pts) - 1):
            if not self._segment_is_clear(arc_pts[i], arc_pts[i + 1]):
                return False
        return True

    def _sphere_surface_path_exists(
        self,
        center: Tuple[float, float, float],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        radius: float,
        n_dirs: int = 48,
    ) -> bool:
        """Check if a collision-free path exists constrained to one sphere surface."""
        if radius < 1e-9:
            return False

        if self._arc_path_is_clear(center, start, end):
            return True

        dirs = self._fibonacci_unit_dirs(n_dirs)
        points: List[Tuple[float, float, float]] = [start, end]
        adj: List[Set[int]] = [set(), set()]
        sample_to_idx: Dict[int, int] = {}

        for di, u in enumerate(dirs):
            p = (
                center[0] + u[0] * radius,
                center[1] + u[1] * radius,
                center[2] + u[2] * radius,
            )
            if self._point_strictly_inside_obstacle(p):
                continue
            sample_to_idx[di] = len(points)
            points.append(p)
            adj.append(set())

        if not sample_to_idx:
            return False

        def add_edge(i: int, j: int) -> None:
            if i == j or j in adj[i]:
                return
            if self._arc_path_is_clear(center, points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)

        k_dir_neighbors = 8
        for di, i_idx in sample_to_idx.items():
            scored: List[Tuple[float, int]] = []
            ui = dirs[di]
            for dj in sample_to_idx.keys():
                if dj == di:
                    continue
                uj = dirs[dj]
                scored.append((ui[0] * uj[0] + ui[1] * uj[1] + ui[2] * uj[2], dj))
            scored.sort(reverse=True)
            for _, dj in scored[:k_dir_neighbors]:
                add_edge(i_idx, sample_to_idx[dj])

        def connect_endpoint(endpoint_idx: int) -> None:
            ep = points[endpoint_idx]
            ev = (ep[0] - center[0], ep[1] - center[1], ep[2] - center[2])
            er = self._norm3(ev)
            if er < 1e-9:
                return
            eu = (ev[0] / er, ev[1] / er, ev[2] / er)

            ranked: List[Tuple[float, int]] = []
            for di, idx in sample_to_idx.items():
                u = dirs[di]
                dot = eu[0] * u[0] + eu[1] * u[1] + eu[2] * u[2]
                ranked.append((dot, idx))
            ranked.sort(reverse=True)

            added = 0
            for _, idx in ranked[:20]:
                add_edge(endpoint_idx, idx)
                if idx in adj[endpoint_idx]:
                    added += 1
                if added >= 6:
                    break

        connect_endpoint(0)
        connect_endpoint(1)

        q: deque[int] = deque([0])
        seen: Set[int] = {0}
        while q:
            u = q.popleft()
            if u == 1:
                return True
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                q.append(v)
        return False

    def _spherical_shell_path_exists(
        self,
        center: Tuple[float, float, float],
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        max_radius_samples: int = 9,
    ) -> bool:
        """
        Check whether a collision-free path exists inside the spherical shell around
        `center` bounded by radii |start-center| and |end-center|.

        This uses a lightweight sampled roadmap in the shell and graph search,
        so the path can be any polyline in the shell (not restricted to a fixed
        radial+arc+radial template).
        """
        v1 = (start[0] - center[0], start[1] - center[1], start[2] - center[2])
        v2 = (end[0] - center[0], end[1] - center[1], end[2] - center[2])
        r1 = self._norm3(v1)
        r2 = self._norm3(v2)
        if r1 < 1e-9 or r2 < 1e-9:
            return False

        if abs(r1 - r2) <= 1e-3:
            return self._sphere_surface_path_exists(center, start, end, radius=0.5 * (r1 + r2))

        r_min = min(r1, r2)
        r_max = max(r1, r2)
        if self._segment_in_shell_is_clear(start, end, center, r_min, r_max):
            return True

        # More radius spread -> denser radial sampling, but keep it lightweight.
        n_r = min(max_radius_samples, max(3, int(math.ceil((r_max - r_min) / 0.2)) + 1))
        n_dirs = 36
        dirs = self._fibonacci_unit_dirs(n_dirs)

        # Precompute nearby direction neighbors by angular similarity.
        k_dir_neighbors = 6
        dir_neighbors: List[List[int]] = []
        for i in range(n_dirs):
            scored: List[Tuple[float, int]] = []
            di = dirs[i]
            for j in range(n_dirs):
                if i == j:
                    continue
                dj = dirs[j]
                scored.append((di[0] * dj[0] + di[1] * dj[1] + di[2] * dj[2], j))
            scored.sort(reverse=True)
            dir_neighbors.append([j for _, j in scored[:k_dir_neighbors]])

        radius_levels = [r_min + (r_max - r_min) * (i / max(1, n_r - 1)) for i in range(n_r)]

        points: List[Tuple[float, float, float]] = [start, end]
        adj: List[Set[int]] = [set(), set()]

        sample_idx: Dict[Tuple[int, int], int] = {}
        for ri, r in enumerate(radius_levels):
            for di, u in enumerate(dirs):
                p = (center[0] + u[0] * r, center[1] + u[1] * r, center[2] + u[2] * r)
                if self._point_strictly_inside_obstacle(p):
                    continue
                sample_idx[(ri, di)] = len(points)
                points.append(p)
                adj.append(set())

        def add_edge(i: int, j: int) -> None:
            if i == j or j in adj[i]:
                return
            if self._segment_in_shell_is_clear(points[i], points[j], center, r_min, r_max):
                adj[i].add(j)
                adj[j].add(i)

        # Connect radial neighbors between adjacent radius levels.
        for ri in range(n_r - 1):
            for di in range(n_dirs):
                a = sample_idx.get((ri, di))
                b = sample_idx.get((ri + 1, di))
                if a is not None and b is not None:
                    add_edge(a, b)

        # Connect tangential neighbors on each radius level.
        for ri in range(n_r):
            for di in range(n_dirs):
                a = sample_idx.get((ri, di))
                if a is None:
                    continue
                for dj in dir_neighbors[di]:
                    b = sample_idx.get((ri, dj))
                    if b is not None:
                        add_edge(a, b)

        def connect_endpoint(endpoint_idx: int) -> None:
            ep = points[endpoint_idx]
            # Prefer local links; fallback to nearest valid links if needed.
            base_radius = max(0.8, 0.6 + 0.3 * (r_max - r_min))
            near: List[int] = []
            ranked: List[Tuple[float, int]] = []
            for j in range(2, len(points)):
                d = self._norm3((points[j][0] - ep[0], points[j][1] - ep[1], points[j][2] - ep[2]))
                ranked.append((d, j))
                if d <= base_radius:
                    near.append(j)

            for j in near:
                add_edge(endpoint_idx, j)

            if adj[endpoint_idx]:
                return

            ranked.sort(key=lambda x: x[0])
            added = 0
            for _, j in ranked[:16]:
                add_edge(endpoint_idx, j)
                if j in adj[endpoint_idx]:
                    added += 1
                if added >= 4:
                    break

        connect_endpoint(0)
        connect_endpoint(1)

        # Graph search for any collision-free shell path.
        q: deque[int] = deque([0])
        seen: Set[int] = {0}
        while q:
            u = q.popleft()
            if u == 1:
                return True
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                q.append(v)

        return False

    def _transition_reachable_via_current_center(self, prev_node: Node, cur_node: Node, next_node: Node) -> bool:
        """
        Check whether a collision-free trajectory exists from prev_node to next_node
        in the spherical shell centered at cur_node's face midpoint.
        """
        c = self.node_to_face_midpoint(cur_node)
        p0 = self.node_to_face_midpoint(prev_node)
        p1 = self.node_to_face_midpoint(next_node)
        return self._spherical_shell_path_exists(c, p0, p1)

    def available_face_dirs(self, free_v: Tuple[int, int, int]) -> List[str]:
        """Directions from free_v center toward adjacent obstacle surfaces."""
        if not self.in_bounds(free_v) or self.occ_at(free_v) != 0:
            return []

        dirs = []
        for f in FACES:
            obs_v = self.addv(free_v, NORM[f])
            if self.in_bounds(obs_v) and self.occ_at(obs_v) == 1:
                dirs.append(f)
        return dirs

    # --- node validity: exposed face of an obstacle voxel ---
    def valid_node(self, node: Node) -> bool:
        if node.face_dir not in FACES:
            return False
        u = self.center_to_idx(node.pos)
        if not self.in_bounds(u):
            return False
        if self.occ_at(u) != 0:
            return False
        obs_v = self.addv(u, NORM[node.face_dir])
        return self.in_bounds(obs_v) and self.occ_at(obs_v) == 1

    # --- heuristic (safe lower bound) ---
    def h(self, a: Node, b: Node) -> int:
        va, vb = self.center_to_idx(a.pos), self.center_to_idx(b.pos)
        return abs(va[0]-vb[0]) + abs(va[1]-vb[1]) + abs(va[2]-vb[2])

    def key(self, s: Node):
        gs = self.g.get(s, INF)
        rs = self.rhs.get(s, INF)
        m = min(gs, rs)
        return (m + self.h(self.start, s) + self.km, m)

    def _node_from_idx_and_face(self, v: Tuple[int, int, int], face: str) -> Node:
        return Node(self.idx_to_center(v), face)

    def _candidate_face_nodes(self, free_v: Tuple[int, int, int]) -> List[Node]:
        """All face nodes on a free voxel, including potentially stale ones for cleanup."""
        if not self.in_bounds(free_v) or self.occ_at(free_v) != 0:
            return []
        return [self._node_from_idx_and_face(free_v, f) for f in FACES]

    def _is_parallel_face_step(self, a: Node, b: Node) -> bool:
        """Relaxed rule: parallel faces with unit normal offset and close in-plane projection."""
        if a == b or not self.valid_node(b):
            return False

        n1 = NORM[a.face_dir]
        n2 = NORM[b.face_dir]
        # Require exactly the same face direction.
        if a.face_dir != b.face_dir:
            return False

        p1 = self.node_to_face_midpoint(a)
        p2 = self.node_to_face_midpoint(b)
        d = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

        # Height difference along the normal direction must be exactly 1.
        dn = d[0] * n1[0] + d[1] * n1[1] + d[2] * n1[2]
        if abs(abs(dn) - 1.0) > 1e-9:
            return False

        # Distance between projected points on the face plane must be in (0, 2).
        tang = (
            d[0] - dn * n1[0],
            d[1] - dn * n1[1],
            d[2] - dn * n1[2],
        )
        proj_dist = math.sqrt(tang[0] * tang[0] + tang[1] * tang[1] + tang[2] * tang[2])
        return 0.0 < proj_dist < 2.0

    # --- successors on the surface graph ---
    def successors(self, node: Node) -> List[Node]:
        if not self.valid_node(node):
            return []

        u = self.center_to_idx(node.pos)
        out: List[Node] = []

        # Node is a face, so expansion starts from this exact face.
        d_face = node.face_dir
        if d_face not in self.available_face_dirs(u):
            return out

        # Step type 1: slide while keeping the same face.
        for step_d in EDGE_DIRS[d_face]:
            u2 = self.addv(u, step_d)
            if not self.in_bounds(u2) or self.occ_at(u2) != 0:
                continue
            s2 = self._node_from_idx_and_face(u2, d_face)
            if self.valid_node(s2) and s2 not in self.visited_faces:
                out.append(s2)

        # Step type 2: edge-flip to an orthogonal face around the same obstacle voxel.
        obs_v = self.addv(u, NORM[d_face])
        if self.in_bounds(obs_v) and self.occ_at(obs_v) == 1:
            n1 = NORM[d_face]
            for f2 in FACES:
                if f2 == d_face:
                    continue
                n2 = NORM[f2]
                if n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] != 0:
                    continue  # only orthogonal faces share a right-angle edge

                u2 = (obs_v[0] - n2[0], obs_v[1] - n2[1], obs_v[2] - n2[2])
                if not self.in_bounds(u2) or self.occ_at(u2) != 0:
                    continue
                s2 = self._node_from_idx_and_face(u2, f2)
                if self.valid_node(s2) and s2 not in self.visited_faces:
                    out.append(s2)

        # Step type 3 (relaxed): jump to a nearby parallel face if geometric constraints match.
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    v2 = (u[0] + dx, u[1] + dy, u[2] + dz)
                    if not self.in_bounds(v2) or self.occ_at(v2) != 0:
                        continue
                    for f2 in FACES:
                        s2 = self._node_from_idx_and_face(v2, f2)
                        if s2 in self.visited_faces:
                            continue
                        if self._is_parallel_face_step(node, s2):
                            out.append(s2)

        # Optional: deduplicate (important when many candidates)
        # keep stable order
        seen: Set[Node] = set()
        dedup = []
        for s in out:
            if s not in seen:
                seen.add(s)
                dedup.append(s)
        return dedup

    def cost(self, a: Node, b: Node) -> int:
        # all moves cost 1 (your requirement)
        # If b is invalid, treat as blocked
        if not self.valid_node(b):
            return INF
        # Disallow switching faces at the same free-voxel center.
        if a.pos == b.pos:
            return INF
        return 1

    # --- D* Lite core routines ---
    def update_vertex(self, u: Node):
        if u != self.goal:
            best = INF
            for s in self.successors(u):
                c = self.cost(u, s)
                if c >= INF: 
                    continue
                best = min(best, c + self.g.get(s, INF))
            self.rhs[u] = best

        if self.g.get(u, INF) != self.rhs.get(u, INF):
            self.OPEN.push(self.key(u), u)

    def compute_shortest_path(self):
        while True:
            top_key = self.OPEN.peek_key()
            start_key = self.key(self.start)
            #print(f"OPEN top key: {top_key}, start key: {start_key}, g(start): {self.g.get(self.start, INF)}, rhs(start): {self.rhs.get(self.start, INF)}")
            if not (top_key < start_key or self.rhs.get(self.start, INF) != self.g.get(self.start, INF)):
                break

            k_old, u = self.OPEN.pop_valid(self.key)
            if u is None:
                break

            if k_old < self.key(u):
                self.OPEN.push(self.key(u), u)
                continue

            gu = self.g.get(u, INF)
            ru = self.rhs.get(u, INF)
            #print(f"Processing node {u} with g={gu}, rhs={ru}")

            # lazy queue: skip entries for nodes that are already consistent
            if gu == ru:
                continue

            if gu > ru:
                self.g[u] = ru
                for p in self.local_predecessors(u):
                    #print(f"gu > ru: Updating predecessor {p} of {u}")
                    self.update_vertex(p)
            else:
                self.g[u] = INF
                self.update_vertex(u)
                for p in self.local_predecessors(u):
                    #print(f"gu < ru: Updating predecessor {p} of {u}")
                    self.update_vertex(p)

    def local_predecessors(self, node: Node) -> List[Node]:
        """
        Exact predecessors would require reverse edges on the surface graph.
        Practical approach for voxel surface graph: update a local candidate set.
        Since updates are 'occasional', local scanning is acceptable.
        """
        v = self.center_to_idx(node.pos)
        cand: List[Node] = []

        # Search a local 3x3x3 free-voxel neighborhood to capture slide + edge-flip predecessors.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    v2 = (v[0] + dx, v[1] + dy, v[2] + dz)
                    if not self.in_bounds(v2) or self.occ_at(v2) != 0:
                        continue
                    for f2 in self.available_face_dirs(v2):
                        c = Node(self.idx_to_center(v2), f2)
                        if self.valid_node(c):
                            cand.append(c)

        # Keep only true predecessors according to current successor model.
        out = [c for c in cand if node in self.successors(c)]
        return list(dict.fromkeys(out))

    def _best_successor(self, node: Node, prev_node: Optional[Node] = None) -> Optional[Node]:
        best = INF
        best_s = None
        for successor in self.successors(node):
            if successor.pos == node.pos:
                continue
            if prev_node is not None and not self._transition_reachable_via_current_center(prev_node, node, successor):
                continue
            value = self.cost(node, successor) + self.g.get(successor, INF)
            if value < best:
                best = value
                best_s = successor
        return best_s

    def _affected_free_voxels(self, voxels) -> Set[Tuple[int, int, int]]:
        affected = set()
        for voxel in voxels:
            affected.add(voxel)
            for dv in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                neighbor = self.addv(voxel, dv)
                if self.in_bounds(neighbor):
                    affected.add(neighbor)
        return affected

    def _update_affected_nodes(self, voxels) -> None:
        affected_nodes: Set[Node] = set()
        for voxel in self._affected_free_voxels(voxels):
            if self.occ_at(voxel) != 0:
                continue
            affected_nodes.update(self._candidate_face_nodes(voxel))

        for node in affected_nodes:
            if self.valid_node(node):
                self.update_vertex(node)
            else:
                self.rhs[node] = INF
                self.g[node] = INF
                self.update_vertex(node)

    # --- usage helpers ---
    def plan_from_current(self) -> bool:
        """Run/repair plan for current start; returns True if reachable."""
        self.compute_shortest_path()
        return self.g.get(self.start, INF) < INF

    def extract_path_stateless(self, max_steps: int = 1000) -> List[Node]:
        """
        Extract path from start to goal WITHOUT modifying planner state.
        This is useful for visualization after replanning.
        
        Returns path by tracing greedy policy from g-values.
        """
        path: List[Node] = []
        if self.g.get(self.start, INF) >= INF:
            return path
        
        current = self.start
        prev: Optional[Node] = None
        path.append(current)
        
        for _ in range(max_steps):
            if current == self.goal:
                break

            best_s = self._best_successor(current, prev_node=prev)
            if best_s is None:
                break

            prev = current
            current = best_s
            path.append(current)
        
        return path
    
    def plan(self, max_steps: int = 1000) -> List[Node]:
        """Plan from current start to goal and return PATH NODES list."""
        path: List[Node] = []
        if not self.plan_from_current():
            return path

        path.append(self.start)
        prev: Optional[Node] = None
        for _ in range(max_steps):
            if self.start == self.goal:
                break

            ns = self.next_step(prev_node=prev)
            if ns is None:
                break

            old_start = self.start
            self.move_start_to(ns)
            path.append(self.start)
            prev = old_start

        return path

    def print_path_nodes(self, path: List[Node]) -> None:
        """Print detailed path information including nodes, positions, and goal status."""
        if not path:
            print("Path is empty.")
            return

        print("\n--- PATH NODES (voxel, face) ---")
        for i, n in enumerate(path):
            print(f"{i:03d}: {n}")

        print("\n--- PATH POINTS (face centers) ---")
        for i, n in enumerate(path):
            print(f"{i:03d}: {self.node_to_face_midpoint(n)}")

        print(f"\nTotal steps: {len(path)}, Hit goal: {self.start == self.goal}")

    def plot_3d_voxels_and_path(self, path: List[Node], title: str = "D* Lite Surface Path") -> None:
        """Visualize 3D path on voxel grid with obstacles, path, start and goal."""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            print("Warning: matplotlib or numpy not available for visualization.")
            return

        # Convert occupancy grid to boolean
        occ_np = np.array(self.occ, dtype=bool)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)

        # --- draw obstacles ---
        ax.voxels(occ_np, alpha=0.5)

        # --- draw path as line through face midpoints ---
        if path:
            pts = np.array([self.node_to_face_midpoint(n) for n in path], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])
            
            # Draw only intermediate path points (exclude start and goal)
            if len(path) > 2:
                pts_mid = np.array([self.node_to_face_midpoint(n) for n in path[1:-1]], dtype=float)
                ax.scatter(pts_mid[:, 0], pts_mid[:, 1], pts_mid[:, 2], s=20, alpha=0.6)

        # --- mark original start/goal at face midpoints ---
        sx, sy, sz = self.node_to_face_midpoint(self.start_orig)
        gx, gy, gz = self.node_to_face_midpoint(self.goal_orig)
        ax.scatter([sx], [sy], [sz], s=80, marker="o", label="start", color="green")  # start
        ax.scatter([gx], [gy], [gz], s=80, marker="^", label="goal", color="red")  # goal

        # axes limits / labels
        ax.set_xlim(0, self.X)
        ax.set_ylim(0, self.Y)
        ax.set_zlim(0, self.Z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        plt.show()

    def next_step(self, prev_node: Optional[Node] = None) -> Optional[Node]:
        """Greedy one-step on surface graph using g-values (like extracting policy)."""
        return self._best_successor(self.start, prev_node=prev_node)

    def move_start_to(self, new_start: Node):
        """Advance the robot along the surface."""
        if not self.valid_node(new_start):
            raise ValueError(f"new_start {new_start} not valid.")
        old = self.start
        self.start = new_start
        self.km += self.h(old, new_start)
        self.visited_faces.add(new_start)  # book-keeping, not required for algorithm

        self.compute_shortest_path()

    def update_voxel(self, v: Tuple[int,int,int], new_occ: int, deferred: bool = False):
        """
        Update occupancy at voxel v.
        
        Args:
            v: voxel coordinate (x, y, z)
            new_occ: new occupancy value (0/1)
            deferred: if True, buffer the update and don't replan immediately;
                      if False, apply immediately and replan
        """
        if deferred:
            # Buffer the update for batch processing
            self.update_buffer[v] = new_occ
            return
        
        # Immediate update (legacy behavior or final application)
        x, y, z = v
        self.occ[x, y, z] = new_occ

        self._update_affected_nodes((v,))

        self.compute_shortest_path()
    
    def buffer_update(self, v: Tuple[int,int,int], new_occ: int):
        """
        Buffer a single voxel occupancy update. 
        Does NOT trigger replanning - use apply_batch_updates() for that.
        
        Args:
            v: voxel coordinate
            new_occ: new occupancy value
        """
        self.update_buffer[v] = new_occ
    
    def buffer_updates(self, updates: Dict[Tuple[int,int,int], int]):
        """
        Buffer multiple voxel updates at once.
        
        Args:
            updates: dict mapping voxel coordinates to new occupancy values
        """
        self.update_buffer.update(updates)
    
    def apply_batch_updates(self, replan: bool = True):
        """
        Apply all buffered voxel updates and optionally trigger replanning.
        This is efficient because it avoids redundant recompute_shortest_path calls.
        
        Args:
            replan: if True, call compute_shortest_path once after all updates;
                    if False, updates are applied but path not recomputed (you must call later)
        
        Returns:
            Number of updates applied
        """
        if not self.update_buffer:
            return 0
        
        num_updates = len(self.update_buffer)
        
        # First pass: apply occupancy changes directly (without replan)
        for v, new_occ in self.update_buffer.items():
            x, y, z = v
            self.occ[x, y, z] = new_occ
        
        # Second pass: update all affected planning nodes.
        self._update_affected_nodes(self.update_buffer.keys())
        
        # Fifth pass: single replanning after all updates
        if replan:
            self.compute_shortest_path()
            print(f"Applied {num_updates} batch updates and replanned")
        else:
            print(f"Applied {num_updates} batch updates (deferred replanning)")
        
        # Clear buffer
        self.update_buffer.clear()
        self.affected_nodes_cache.clear()
        
        return num_updates
    
    def clear_buffer(self):
        """Clear buffered updates without applying them."""
        self.update_buffer.clear()
        self.affected_nodes_cache.clear()
    
    def get_buffer_size(self) -> int:
        """Get number of pending updates in buffer."""
        return len(self.update_buffer)


# --- Example of how you'd call it (you can delete this in your project) ---
# 

if __name__ == "__main__":
    import numpy as np
    import random

    # ---------- build a demo 3D voxel world ----------
    X, Y, Z = 5, 5, 5
    occ = np.zeros((X, Y, Z), dtype=np.uint8)

    # Example obstacle block
    for x in range(0, X):
        for y in range(0, Y):
            z = random.randint(1, Z-2)
            occ[x, y, 0:z] = 1

    # Start/Goal are two valid face nodes: Node((free-cell-center), face_dir)
    valid_faces = []
    for x in range(X):
        for y in range(Y):
            for zc in range(Z):
                if occ[x, y, zc] != 0:
                    continue
                for f in FACES:
                    ox, oy, oz = x + NORM[f][0], y + NORM[f][1], zc + NORM[f][2]
                    if 0 <= ox < X and 0 <= oy < Y and 0 <= oz < Z and occ[ox, oy, oz] == 1:
                        valid_faces.append(Node((x + 0.5, y + 0.5, zc + 0.5), f))

    if len(valid_faces) < 2:
        raise RuntimeError("Not enough valid faces to sample start and goal.")

    # Keep generation style but explicitly sample two distinct faces.
    while True:
        start, goal = random.sample(valid_faces, 2)
        if start != goal:
            break
    print(f"Random start={start}, goal={goal}")

    planner = DStarLiteSurface3D(occ, (X, Y, Z), start, goal)

    path = planner.plan(max_steps=20)
    ok = len(path) > 0
    print("reachable:", ok)

    if ok:
        planner.print_path_nodes(path)

    # ---------- 3D visualization ----------
    planner.plot_3d_voxels_and_path(path, title="D* Lite on Surface Graph (3D)")


# ============================================================================
# BATCH REPLANNING API DOCUMENTATION
# ============================================================================
"""
D* Lite now supports efficient batch updates for dynamic replanning:

BASIC API:
    planner = DStarLiteSurface3D(occ, size_xyz, start, goal)
    
    # Method 1: Buffer single update
    planner.buffer_update((x,y,z), new_occ_value)
    
    # Method 2: Buffer multiple updates at once
    updates_dict = {(x1,y1,z1): 0, (x2,y2,z2): 1, ...}
    planner.buffer_updates(updates_dict)
    
    # Apply all buffered updates with single replan (efficient!)
    num_applied = planner.apply_batch_updates(replan=True)
    
    # Extract path after replanning
    new_path = planner.plan(max_steps=100)

PERFORMANCE BENEFITS:
    Without batching:
        for each voxel change:
            planner.update_voxel(v, new_occ)  # calls compute_shortest_path() internally
        Total: N calls to compute_shortest_path() for N voxel changes
    
    With batching:
        for each voxel change:
            planner.buffer_update(v, new_occ)  # No planning yet
        planner.apply_batch_updates(replan=True)  # Single compute_shortest_path() call!
        Total: 1 call to compute_shortest_path() for N voxel changes

DEFERRED REPLANNING PATTERN:
    # Collect updates without immediate replanning
    changes = detect_map_changes()  # returns dict
    planner.buffer_updates(changes)
    planner.apply_batch_updates(replan=False)  # Apply updates but defer replanning
    
    # Later, when ready to plan:
    planner.compute_shortest_path()  # Replanning done
    path = planner.plan()

UTILITY METHODS:
    planner.get_buffer_size()        # Check how many updates are buffered
    planner.clear_buffer()           # Discard all buffered updates
    planner.update_voxel(v, occ, deferred=False)  # Single immediate update (legacy)
"""
