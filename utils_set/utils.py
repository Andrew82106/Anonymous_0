"""
工具模块 (Utilities)
提供配置加载等通用功能
"""

import yaml
from pathlib import Path
from typing import Any, Optional

class ConfigLoader:
    """
    配置文件加载器，支持嵌套键访问
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 默认加载 llms/config.yaml
            project_root = Path(__file__).parent
            config_path = project_root / 'llms' / 'config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """
        加载 YAML 配置文件
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的嵌套键
        
        Example:
        --------
        config.get('models.text_models')  # 返回 text_models 列表
        config.get('models.text_models.0.name')  # 返回第一个模型的名称
        
        Parameters:
        -----------
        key_path : str
            点分隔的键路径，如 'models.text_models'
        default : Any
            如果键不存在，返回的默认值
        
        Returns:
        --------
        Any : 配置值或默认值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list):
                try:
                    index = int(key)
                    value = value[index] if 0 <= index < len(value) else None
                except (ValueError, IndexError):
                    value = None
            else:
                value = None
            
            if value is None:
                return default
        
        return value
    
    def reload(self):
        """
        重新加载配置文件
        """
        self.config = self._load_config()

# 全局配置加载器实例
config_loader = ConfigLoader()
