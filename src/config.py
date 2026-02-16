"""
配置模块 - 加载并提供全局配置参数
"""
import yaml
import os

# 获取配置文件路径
_config_path = os.path.join(os.path.dirname(__file__), "../config/default.yaml")

# 加载配置
with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# 导出常用配置项
scaling_factor = _cfg["sim"]["scaling_factor"]
grid_size = _cfg["sim"]["grid_size"]
delta_per_step = _cfg["sim"]["delta_per_step"]
switch_period_s = _cfg["sim"]["switch_period_s"]

# 如果需要访问完整配置
cfg = _cfg
