"""
通用工具函数模块
"""

def get_object_path(cfg, name):
    """
    从配置文件的 objects 列表中获取指定对象的路径
    
    Args:
        cfg: 配置字典
        name: 对象名称
        
    Returns:
        str: 对象路径
        
    Raises:
        ValueError: 找不到指定对象时抛出
    """
    for obj in cfg.get("objects", []):
        if obj.get("name") == name:
            return obj["path"]
    raise ValueError(f"Object '{name}' not found in config")


def get_all_robot_paths(cfg):
    """
    获取所有机器人对象的路径字典
    
    Args:
        cfg: 配置字典
        
    Returns:
        dict: {name: path} 格式的字典
    """
    return {obj["name"]: obj["path"] for obj in cfg.get("objects", []) if obj.get("type") == "robot"}

def get_all_block_paths(cfg):
    """
    获取所有块对象的路径字典
    
    Args:
        cfg: 配置字典
        
    Returns:
        dict: {name: path} 格式的字典
    """
    return {obj["name"]: obj["path"] for obj in cfg.get("objects", []) if obj.get("type") == "block"}

def get_all_cuboid_paths(cfg):
    """
    获取所有立方体对象的路径字典
    
    Args:
        cfg: 配置字典
        
    Returns:
        dict: {name: path} 格式的字典
    """
    return {obj["name"]: obj["path"] for obj in cfg.get("objects", []) if obj.get("type") == "cuboid"}


def get_objects_by_type(cfg, obj_type):
    """
    按类型筛选对象
    
    Args:
        cfg: 配置字典
        obj_type: 对象类型（如 "robot"）
        
    Returns:
        list: 匹配的对象列表
    """
    return [obj for obj in cfg.get("objects", []) if obj.get("type") == obj_type]
