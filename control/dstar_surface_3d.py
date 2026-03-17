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

# For each face, the 4 in-plane edge directions (orthogonal to the face normal)
EDGE_DIRS = {
    "+X": [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "-X": [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "+Y": [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
    "-Y": [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
    "+Z": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    "-Z": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
}

Node = Tuple[Tuple[int, int, int], str]  # ((x,y,z), face)


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

    # --- node validity: exposed face of an obstacle voxel ---
    def valid_node(self, node: Node) -> bool:
        v, f = node
        if not self.in_bounds(v): return False
        if self.occ_at(v) != 1: return False
        u = self.addv(v, NORM[f])
        if not self.in_bounds(u): return False
        return self.occ_at(u) == 0  # exposed

    # --- heuristic (safe lower bound) ---
    def h(self, a: Node, b: Node) -> int:
        (va, _), (vb, _) = a, b
        return abs(va[0]-vb[0]) + abs(va[1]-vb[1]) + abs(va[2]-vb[2])

    def key(self, s: Node):
        gs = self.g.get(s, INF)
        rs = self.rhs.get(s, INF)
        m = min(gs, rs)
        return (m + self.h(self.start, s) + self.km, m)

    # --- successors on the surface graph (1B/2A) ---
    def successors(self, node: Node) -> List[Node]:
        v, f = node
        out: List[Node] = []

        # A) turn on the same obstacle voxel around an edge: to orthogonal faces (4 candidates)
        for f2 in FACES:
            if f2 == f: 
                continue
            # orthogonal: dot(n(f), n(f2)) == 0
            n1 = NORM[f]
            n2 = NORM[f2]
            if n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2] != 0:
                continue
            nnode = (v, f2)
            adr = self.addv(v, NORM[f2])  # adjacent free voxel in direction of new face
            if self.valid_node(nnode) and not self.visited[adr[0]][adr[1]][adr[2]]:  # optional: prefer already visited faces to reduce branching
                out.append(nnode)

        # B) switch to another obstacle face via free-layer edge move

        u = self.addv(v, NORM[f])      # free voxel adjacent to current face
        # if not self.in_bounds(u) or self.occ_at(u) != 0:
        #     return out  # should not happen if valid_node, but keep safe

        for d in EDGE_DIRS[f]:
            u2 = self.addv(u, d)
            if not self.in_bounds(u2): 
                continue
            if self.occ_at(u2) != 0:
                continue  # must stay in free layer while sliding

            # At u2, we can "attach" to any obstacle voxel adjacent to u2
            # by choosing a face f2 whose outward neighbor is u2.
            for f2 in FACES:
                v2 = self.addv(u2, (-NORM[f2][0], -NORM[f2][1], -NORM[f2][2]))  # v2 = u2 - n(f2)
                nnode = (v2, f2)
                adr = self.addv(v2, NORM[f2])  # adjacent free voxel in direction of new face
                if self.valid_node(nnode) and not self.visited[adr[0]][adr[1]][adr[2]]:  # optional: prefer already visited faces to reduce branching
                    out.append(nnode)

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
        return 1 if self.valid_node(b) else INF

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
        (v, f) = node
        cand: List[Node] = []

        # candidates on same voxel (all faces)
        for f2 in FACES:
            cand.append((v, f2))

        # candidates from neighboring obstacle voxels around v (6 neighbors, all faces)
        for dv in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            v2 = self.addv(v, dv)
            if self.in_bounds(v2):
                for f2 in FACES:
                    cand.append((v2, f2))

        # filter valid
        out = [c for c in cand if self.valid_node(c)]
        # dedup
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
        v,f = new_start
        adr = self.addv(v, NORM[f])
        self.visited[v[0]][v[1]][v[2]] = True  # book-keeping, not required for algorithm
         # adjacent free voxel

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

        # update all faces on these voxels (if valid)
        affected_nodes: Set[Node] = set()
        for vv in affected_voxels:
            for f in FACES:
                n = (vv, f)
                if self.valid_node(n) or n in self.g or n in self.rhs:
                    affected_nodes.add(n)

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
        """Return face center point (x,y,z) in voxel coordinates."""
        (x, y, z), f = node
        nx, ny, nz = NORM[f]
        return (x + 0.5 + 0.5 * nx, y + 0.5 + 0.5 * ny, z + 0.5 + 0.5 * nz)

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

    # Start/Goal are (obstacle voxel, exposed face)
    valid_nodes = []
    for x in range(X):
        for y in range(Y):
            for zc in range(Z):
                if occ[x, y, zc] != 1:
                    continue
                for f in FACES:
                    nx, ny, nz = NORM[f]
                    ux, uy, uz = x + nx, y + ny, zc + nz
                    if 0 <= ux < X and 0 <= uy < Y and 0 <= uz < Z and occ[ux, uy, uz] == 0:
                        valid_nodes.append(((x, y, zc), f))

    if len(valid_nodes) < 2:
        raise RuntimeError("Not enough valid nodes to sample start and goal.")

    start, goal = random.sample(valid_nodes, 2)
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