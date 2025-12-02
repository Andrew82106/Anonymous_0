"""
工具模块 (Utilities)
提供配置加载、路径管理等通用功能
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional

class ConfigLoader:
    """
    配置文件加载器，支持嵌套键访问
    """
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # 使用 PathConfig 获取默认配置文件路径（需要延迟导入避免循环依赖）
            # 这里直接计算，避免循环依赖
            project_root = Path(__file__).parent.parent
            config_path = os.path.join(project_root, 'llms', 'config.yaml')
        
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


class PathConfig:
    """
    项目路径配置类
    统一管理所有项目路径，支持跨平台
    """
    def __init__(self):
        # 项目根目录：从当前文件 (utils_set/utils.py) 向上两级
        self._project_root = Path(__file__).parent.parent.resolve()
        
        # 确保使用绝对路径，避免相对路径问题
        self._ensure_absolute_paths()
    
    def _ensure_absolute_paths(self):
        """确保所有路径都是绝对路径"""
        if not self._project_root.is_absolute():
            self._project_root = self._project_root.resolve()
    
    # ==================== 根目录 ====================
    @property
    def project_root(self) -> Path:
        """项目根目录"""
        return self._project_root
    
    # ==================== 主要文件夹 ====================
    @property
    def background_dir(self) -> Path:
        """背景文档目录"""
        return self._project_root / 'background'
    
    @property
    def results_dir(self) -> Path:
        """结果存储目录"""
        return self._project_root / 'results'
    
    @property
    def tests_dir(self) -> Path:
        """测试脚本目录"""
        return self._project_root / 'tests'
    
    @property
    def utils_dir(self) -> Path:
        """工具模块目录（utils_set）"""
        return self._project_root / 'utils_set'
    
    @property
    def llms_dir(self) -> Path:
        """LLM管理系统目录"""
        return self._project_root / 'llms'
    
    # ==================== 配置文件 ====================
    @property
    def llm_config_file(self) -> Path:
        """LLM配置文件路径"""
        return self.llms_dir / 'config.yaml'
    
    @property
    def task_file(self) -> Path:
        """任务文档路径"""
        return self.background_dir / 'task.md'
    
    # ==================== 结果文件 ====================
    @property
    def experiment_results_file(self) -> Path:
        """合成数据实验结果文件"""
        return self.results_dir / 'experiment_results.json'
    
    @property
    def real_network_results_file(self) -> Path:
        """真实网络测试结果文件"""
        return self.results_dir / 'real_network_results.json'
    
    # ==================== 工具方法 ====================
    def ensure_dir(self, dir_path: Path) -> Path:
        """
        确保目录存在，不存在则创建
        
        Parameters:
        -----------
        dir_path : Path
            目录路径
        
        Returns:
        --------
        Path : 目录路径
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def get_relative_path(self, absolute_path: Path, from_dir: Optional[Path] = None) -> Path:
        """
        获取相对路径
        
        Parameters:
        -----------
        absolute_path : Path
            绝对路径
        from_dir : Optional[Path]
            参照目录，默认为项目根目录
        
        Returns:
        --------
        Path : 相对路径
        """
        if from_dir is None:
            from_dir = self.project_root
        try:
            return Path(absolute_path).relative_to(from_dir)
        except ValueError:
            # 如果路径不在参照目录下，返回绝对路径
            return Path(absolute_path)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"PathConfig(project_root='{self.project_root}')"
    
    def print_all_paths(self):
        """打印所有路径配置（用于调试）"""
        print("="*60)
        print("项目路径配置 (PathConfig)")
        print("="*60)
        print(f"📁 项目根目录: {self.project_root}")
        print(f"\n主要文件夹:")
        print(f"  📂 background/  : {self.background_dir}")
        print(f"  📂 results/     : {self.results_dir}")
        print(f"  📂 tests/       : {self.tests_dir}")
        print(f"  📂 utils_set/   : {self.utils_dir}")
        print(f"  📂 llms/        : {self.llms_dir}")
        print(f"\n配置文件:")
        print(f"  ⚙️  LLM配置     : {self.llm_config_file}")
        print(f"  📝 任务文档     : {self.task_file}")
        print(f"\n结果文件:")
        print(f"  📊 实验结果     : {self.experiment_results_file}")
        print(f"  📊 网络测试结果 : {self.real_network_results_file}")
        print("="*60)


# 全局配置实例
path_config = PathConfig()
config_loader = ConfigLoader()
