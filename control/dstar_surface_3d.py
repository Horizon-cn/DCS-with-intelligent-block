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
        print(f"Initialized D* Lite with start={self.start} and goal={self.goal}")
        
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
        path.append(current)
        
        for _ in range(max_steps):
            if current == self.goal:
                break
            
            # Greedy step: find successor with minimum cost
            best = INF
            best_s = None
            for s in self.successors(current):
                if s.pos == current.pos:
                    continue
                c = self.cost(current, s)
                val = c + self.g.get(s, INF)
                if val < best:
                    best = val
                    best_s = s
            
            if best_s is None:
                break
            
            current = best_s
            path.append(current)
        
        return path
    
    def plan(self, max_steps: int = 1000) -> List[Node]:
        """Plan from current start to goal and return PATH NODES list."""
        path: List[Node] = []
        if not self.plan_from_current():
            return path

        path.append(self.start)
        for _ in range(max_steps):
            if self.start == self.goal:
                break

            ns = self.next_step()
            if ns is None:
                break

            self.move_start_to(ns)
            path.append(self.start)

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
            print(f"{i:03d}: {n.pos}")

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

        # --- draw path as line ---
        if path:
            pts = np.array([n.pos for n in path], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])
            
            # Draw only intermediate path points (exclude start and goal)
            if len(path) > 2:
                pts_mid = np.array([n.pos for n in path[1:-1]], dtype=float)
                ax.scatter(pts_mid[:, 0], pts_mid[:, 1], pts_mid[:, 2], s=20, alpha=0.6)

        # --- mark original start/goal (use start_orig/goal_orig to preserve original positions) ---
        sx, sy, sz = self.start_orig.pos
        gx, gy, gz = self.goal_orig.pos
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
        
        # Second pass: identify all affected surface nodes
        affected_voxels = set()
        for v in self.update_buffer.keys():
            affected_voxels.add(v)
            # Add 6-neighbors since exposure depends on neighborhood
            for dv in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                vv = self.addv(v, dv)
                if self.in_bounds(vv):
                    affected_voxels.add(vv)
        
        # Third pass: update all affected planning nodes
        affected_nodes: Set[Node] = set()
        for vv in affected_voxels:
            if not self.in_bounds(vv):
                continue
            if self.occ_at(vv) != 0:
                continue
            
            # Get current directions
            dirs = self.available_face_dirs(vv)
            for f in dirs:
                affected_nodes.add(Node(self.idx_to_center(vv), f))
            
            # Keep old keys for cleanup
            probe = Node(self.idx_to_center(vv), "+X")
            if probe in self.g or probe in self.rhs:
                affected_nodes.add(probe)
        
        # Fourth pass: update vertices (accumulate affected nodes, don't replan yet)
        for n in affected_nodes:
            if self.valid_node(n):
                self.update_vertex(n)
            else:
                self.rhs[n] = INF
                self.g[n] = INF
                self.update_vertex(n)
        
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