import heapq
import random
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
    """A planning node: free-cell center position + direction toward an adjacent obstacle surface."""
    pos: Tuple[float, float, float]
    face_dir: str

    def __hash__(self):
        # Same position is the same planning point regardless of face direction.
        return hash(self.pos)

    def __eq__(self, other):
        return isinstance(other, Node) and self.pos == other.pos


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
        start/goal: Node=((x,y,z), face) on an obstacle voxel and must be an exposed face.
        """
        self.occ = occ
        self.X, self.Y, self.Z = size_xyz
        self.visited = [[[False for _ in range(self.Z)] for _ in range(self.Y)] for _ in range(self.X)]
        self.start = start
        self.goal = goal

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
        print(f"Initialized D* Lite with start={self.start} and goal={self.goal}")

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

    # --- successors on the surface graph (1B/2A) ---
    def successors(self, node: Node) -> List[Node]:
        u = self.center_to_idx(node.pos)
        out: List[Node] = []

        # At a point, all available face directions can be used for next-step expansion.
        usable_dirs = self.available_face_dirs(u)
        if not usable_dirs:
            return out

        # Slide on free layer along in-plane edge directions of each usable face.
        for d_face in usable_dirs:
            for step_d in EDGE_DIRS[d_face]:
                u2 = self.addv(u, step_d)
                if not self.in_bounds(u2) or self.occ_at(u2) != 0:
                    continue

                for d2 in self.available_face_dirs(u2):
                    if not self.visited[u2[0]][u2[1]][u2[2]]:
                        out.append(Node(self.idx_to_center(u2), d2))

        # Edge flip: move around a right-angle edge of the same obstacle voxel
        # from face d_face to an orthogonal face f2.
        for d_face in usable_dirs:
            obs_v = self.addv(u, NORM[d_face])
            if not self.in_bounds(obs_v) or self.occ_at(obs_v) != 1:
                continue

            n1 = NORM[d_face]
            for f2 in FACES:
                if f2 == d_face:
                    continue
                n2 = NORM[f2]
                if n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] != 0:
                    continue  # only orthogonal faces share a right-angle edge

                # free cell adjacent to same obstacle voxel on face f2
                u2 = (obs_v[0] - n2[0], obs_v[1] - n2[1], obs_v[2] - n2[2])
                if not self.in_bounds(u2) or self.occ_at(u2) != 0:
                    continue
                if f2 in self.available_face_dirs(u2):
                    if not self.visited[u2[0]][u2[1]][u2[2]]:
                        out.append(Node(self.idx_to_center(u2), f2))

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
            print(f"OPEN top key: {top_key}, start key: {start_key}, g(start): {self.g.get(self.start, INF)}, rhs(start): {self.rhs.get(self.start, INF)}")
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
            print(f"Processing node {u} with g={gu}, rhs={ru}")

            # lazy queue: skip entries for nodes that are already consistent
            if gu == ru:
                continue

            if gu > ru:
                self.g[u] = ru
                for p in self.local_predecessors(u):
                    print(f"gu > ru: Updating predecessor {p} of {u}")
                    self.update_vertex(p)
            else:
                self.g[u] = INF
                self.update_vertex(u)
                for p in self.local_predecessors(u):
                    print(f"gu < ru: Updating predecessor {p} of {u}")
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

    # --- usage helpers ---
    def plan_from_current(self) -> bool:
        """Run/repair plan for current start; returns True if reachable."""
        self.compute_shortest_path()
        return self.g.get(self.start, INF) < INF

    def next_step(self) -> Optional[Node]:
        """Greedy one-step on surface graph using g-values (like extracting policy)."""
        best = INF
        best_s = None
        for s in self.successors(self.start):
            if s.pos == self.start.pos:
                continue
            c = self.cost(self.start, s)
            val = c + self.g.get(s, INF)
            if val < best:
                best = val
                best_s = s
        return best_s

    def move_start_to(self, new_start: Node):
        """Advance the robot along the surface."""
        if not self.valid_node(new_start):
            raise ValueError(f"new_start {new_start} not valid.")
        old = self.start
        self.start = new_start
        self.km += self.h(old, new_start)
        u = self.center_to_idx(new_start.pos)
        self.visited[u[0]][u[1]][u[2]] = True  # book-keeping, not required for algorithm

        self.compute_shortest_path()

    def update_voxel(self, v: Tuple[int,int,int], new_occ: int):
        """
        Occasional map update: change occupancy at voxel v (0/1),
        then locally repair affected surface nodes.
        """
        x,y,z = v
        self.occ[x,y,z] = new_occ

        # affected voxels: v and its 6-neighbors (because exposure depends on 6-neighborhood)
        affected_voxels = [v]
        for dv in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            vv = self.addv(v, dv)
            if self.in_bounds(vv):
                affected_voxels.append(vv)

        # update all nearby free-cell points whose available surfaces may change
        affected_nodes: Set[Node] = set()
        for vv in affected_voxels:
            if not self.in_bounds(vv):
                continue
            if self.occ_at(vv) != 0:
                continue
            dirs = self.available_face_dirs(vv)
            for f in dirs:
                affected_nodes.add(Node(self.idx_to_center(vv), f))

            # Keep old keys around for cleanup when a point loses all faces.
            probe = Node(self.idx_to_center(vv), "+X")
            if probe in self.g or probe in self.rhs:
                affected_nodes.add(probe)

        for n in affected_nodes:
            if self.valid_node(n):
                self.update_vertex(n)
            else:
                # If it becomes invalid, set rhs=INF and g=INF to remove its influence
                self.rhs[n] = INF
                self.g[n] = INF
                self.update_vertex(n)

        self.compute_shortest_path()


# --- Example of how you'd call it (you can delete this in your project) ---
# 

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ---------- helpers for visualization ----------
    def face_center(node):
        """Node now stores free-cell center directly."""
        return node.pos

    def plot_3d_voxels_and_path(occ, path_nodes, start, goal, title="D* Lite Surface Path"):
        X, Y, Z = occ.shape
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)

        # --- draw obstacles ---
        filled = occ.astype(bool)
        # ax.voxels expects indexing as [x,y,z] if you pass in the same shaped boolean array
        ax.voxels(filled, alpha=0.5)

        # --- draw path as points (face centers) ---
        if path_nodes:
            pts = np.array([face_center(n) for n in path_nodes], dtype=float)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20)

            # connect with line for clarity
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])

        # --- mark start/goal ---
        sx, sy, sz = face_center(start)
        gx, gy, gz = face_center(goal)
        ax.scatter([sx], [sy], [sz], s=80, marker="o")  # start
        ax.scatter([gx], [gy], [gz], s=80, marker="^")  # goal

        # axes limits / labels
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_zlim(0, Z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()

    # ---------- build a demo 3D voxel world ----------
    X, Y, Z = 5, 5, 5
    occ = np.zeros((X, Y, Z), dtype=np.uint8)

    # Example obstacle block
    for x in range(0, X):
        for y in range(0, Y):
            z = random.randint(1, Z-2)
            occ[x, y, 0:z] = 1

    # Start/Goal are (free-cell center, direction-to-obstacle-face)
    valid_nodes = []
    for x in range(X):
        for y in range(Y):
            for zc in range(Z):
                if occ[x, y, zc] != 0:
                    continue
                for f in FACES:
                    ox, oy, oz = x + NORM[f][0], y + NORM[f][1], zc + NORM[f][2]
                    if 0 <= ox < X and 0 <= oy < Y and 0 <= oz < Z and occ[ox, oy, oz] == 1:
                        valid_nodes.append(Node((x + 0.5, y + 0.5, zc + 0.5), f))

    if len(valid_nodes) < 2:
        raise RuntimeError("Not enough valid nodes to sample start and goal.")

    # Ensure different coordinates (same coordinate is treated as same planning point).
    while True:
        start, goal = random.sample(valid_nodes, 2)
        if start.pos != goal.pos:
            break
    print(f"Random start={start}, goal={goal}")

    planner = DStarLiteSurface3D(occ, (X, Y, Z), start, goal)

    ok = planner.plan_from_current()
    print("reachable:", ok)

    # ---------- extract & print path ----------
    path = []
    if ok:
        path.append(planner.start)

        for step in range(10):
            if planner.start == planner.goal:
                break

            ns = planner.next_step()
            if ns is None:
                print("stuck: no next step")
                break

            # move and record
            planner.move_start_to(ns)
            path.append(planner.start)

        # Print path nodes
        print("\n--- PATH NODES (voxel, face) ---")
        for i, n in enumerate(path):
            print(f"{i:03d}: {n}")

        # Print as face-centers (optional, easier to visualize numerically)
        print("\n--- PATH POINTS (face centers) ---")
        for i, n in enumerate(path):
            print(f"{i:03d}: {face_center(n)}")

        print("\nsteps:", len(path), "hit goal:", planner.start == planner.goal)

    # ---------- 3D visualization ----------
    plot_3d_voxels_and_path(occ, path, start, goal, title="D* Lite on Surface Graph (3D)")