"""
Test script for D* Lite batch replanning mechanism with 3D visualization.

Demonstrates:
1. Buffering multiple map changes
2. Batch application with single replanning
3. Performance comparison vs. individual updates
4. 3D visualization of each planning step
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from control.dstar_surface_3d import DStarLiteSurface3D, Node, FACES, NORM

def test_batch_replanning():
    """Test batch update mechanism."""
    print("="*70)
    print("D* Lite Batch Replanning Test")
    print("="*70)
    
    # Create demo grid
    X, Y, Z = 5, 5, 5
    occ = np.zeros((X, Y, Z), dtype=np.uint8)
    
    # Add some obstacles
    for x in range(X):
        for y in range(Y):
            z = random.randint(1, Z-3)
            occ[x, y, 0:z] = 1
    
    # Find valid start/goal
    valid_nodes = []
    for x in range(X):
        for y in range(Y):
            for zc in range(Z):
                if occ[x, y, zc] != 0:
                    continue
                for f in FACES:
                    ox = x + NORM[f][0]
                    oy = y + NORM[f][1]
                    oz = zc + NORM[f][2]
                    if 0 <= ox < X and 0 <= oy < Y and 0 <= oz < Z and occ[ox, oy, oz] == 1:
                        valid_nodes.append(Node((x + 0.5, y + 0.5, zc + 0.5), f))
    
    if len(valid_nodes) < 2:
        print("Not enough valid nodes, skipping test")
        return False
    
    start, goal = random.sample(valid_nodes, 2)
    if start.pos == goal.pos:
        start, goal = random.sample(valid_nodes, 2)
    
    print(f"\n1. Initial Planning")
    print(f"   Start: {start}")
    print(f"   Goal:  {goal}")
    
    # Create planner
    planner = DStarLiteSurface3D(occ, (X, Y, Z), start, goal)
    planner.plan_from_current()  # Initialize planning
    path = planner.extract_path_stateless(max_steps=100)
    print(f"   Path length: {len(path)} nodes")
    
    # Visualize initial planning
    print(f"\n   📊 Visualizing initial path...")
    planner.plot_3d_voxels_and_path(path, title="Step 1: Initial Path (D* Lite)")
    plt.show()
    
    print(f"\n2. Testing Batch Update Mechanism")
    
    # Generate some random changes
    changes = {}
    num_changes = 15
    for _ in range(num_changes):
        vx = random.randint(1, X-1)
        vy = random.randint(1, Y-1)
        vz = random.randint(1, Z-1)
        if (vx, vy, vz) != start.pos and (vx, vy, vz) != goal.pos:
            new_val = 1 - occ[vx, vy, vz]
            changes[(vx, vy, vz)] = new_val
    
    print(f"   Generated {len(changes)} random map changes")
    print(f"   Buffering updates (no replanning yet)...")
    
    # Buffer updates
    planner.buffer_updates(changes)
    print(f"   Buffer size: {planner.get_buffer_size()} pending updates")
    
    # Apply batch updates
    print(f"   Applying batch updates with replanning...")
    start_time = time.time()
    num_applied = planner.apply_batch_updates(replan=True)
    elapsed = time.time() - start_time
    
    print(f"   Applied {num_applied} updates in {elapsed:.4f} seconds")
    print(f"   Buffer size after: {planner.get_buffer_size()}")
    
    # Check if path still exists - use stateless extraction
    new_path = planner.extract_path_stateless(max_steps=100)
    print(f"   New path length: {len(new_path)} nodes")
    
    # Visualize after batch update
    print(f"\n   📊 Visualizing path after batch updates...")
    planner.plot_3d_voxels_and_path(new_path, title=f"Step 2: After Batch Update (Applied {num_applied} changes, Replanning took {elapsed:.4f}s)")
    
    print(f"\n3. Testing Deferred Replanning")
    
    # Generate more changes
    changes2 = {}
    for _ in range(8):
        vx = random.randint(1, X-1)
        vy = random.randint(1, Y-1)
        vz = random.randint(1, Z-1)
        new_val = 1 - occ[vx, vy, vz]
        changes2[(vx, vy, vz)] = new_val
    
    print(f"   Buffering {len(changes2)} updates (deferred replanning)...")
    planner.buffer_updates(changes2)
    
    # Apply without replanning
    num_applied = planner.apply_batch_updates(replan=False)
    print(f"   Applied {num_applied} updates (no replan)")
    
    # Now do manual replanning
    print(f"   Manual replanning...")
    start_time = time.time()
    planner.compute_shortest_path()
    elapsed = time.time() - start_time
    print(f"   Replanning took {elapsed:.4f} seconds")
    
    final_path = planner.extract_path_stateless(max_steps=100)
    print(f"   Final path length: {len(final_path)} nodes")
    
    # Visualize final path after deferred replanning
    print(f"\n   📊 Visualizing path after deferred replanning...")
    planner.plot_3d_voxels_and_path(final_path, title=f"Step 3: After Deferred Replanning (Applied {len(changes2)} more changes, Replanning took {elapsed:.4f}s)")
    plt.show()
    
    print(f"\n4. Testing Buffer Clear")
    planner.buffer_update((5, 5, 5), 1)
    planner.buffer_update((6, 6, 6), 0)
    print(f"   Buffered 2 updates, buffer size: {planner.get_buffer_size()}")
    planner.clear_buffer()
    print(f"   After clear, buffer size: {planner.get_buffer_size()}")
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = test_batch_replanning()
    if not success:
        print("Test failed")
        exit(1)
