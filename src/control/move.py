def step(sim, obj, axis_index, delta):
    """
    沿指定轴移动对象
    
    Args:
        sim: CoppeliaSim API 对象
        obj: 对象句柄
        axis_index: 轴索引 (0=x, 1=y, 2=z)
        delta: 移动增量
        
    Returns:
        list: 更新后的位置 [x, y, z]
    """
    p = sim.getObjectPosition(obj, -1)
    p[axis_index] += delta
    sim.setObjectPosition(obj, -1, p)
    return p

def goto(sim, obj, target_pos, speed=0.05, tolerance=0.01):
    """
    每次调用向目标位置前进一步（稳定速度）
    
    Args:
        sim: CoppeliaSim API 对象
        obj: 对象句柄
        target_pos: 目标位置 [x, y, z]
        speed: 每步移动的最大距离（速度）
        tolerance: 到达目标的容差（距离阈值）
        
    Returns:
        tuple: (是否到达, 当前位置)
    """
    cur_pos = sim.getObjectPosition(obj, -1)
    direction = [target_pos[i] - cur_pos[i] for i in range(3)]
    distance = sum(d**2 for d in direction) ** 0.5
    
    if distance <= tolerance:
        return (True, cur_pos)
    
    step = [d / distance * min(speed, distance) for d in direction]
    new_pos = [cur_pos[i] + step[i] for i in range(3)]
    sim.setObjectPosition(obj, -1, new_pos)
    
    return (False, new_pos)