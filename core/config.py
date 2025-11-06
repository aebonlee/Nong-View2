"""
Configuration management module for Nong-View2
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file
        
        Args:
            config_path: Path to config.yaml file
        """
        if config_path is None:
            # Get default config path from project root
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file
        
        Returns:
            Dictionary containing configuration
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        base_dir = self.config_path.parent
        
        # Create data directories
        for dir_key in ['input_dir', 'output_dir', 'temp_dir', 'model_dir', 'log_dir']:
            if dir_key in self.get('paths', {}):
                dir_path = base_dir / self.get('paths', {}).get(dir_key, '')
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, *keys, default=None) -> Any:
        """Get configuration value by nested keys
        
        Args:
            *keys: Nested keys to access configuration
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    def set(self, value: Any, *keys) -> None:
        """Set configuration value by nested keys
        
        Args:
            value: Value to set
            *keys: Nested keys to access configuration
        """
        if not keys:
            return
        
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to YAML file
        
        Args:
            path: Path to save configuration (uses original path if None)
        """
        save_path = path if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    @property
    def input_dir(self) -> Path:
        """Get input directory path"""
        return Path(self.config_path.parent) / self.get('paths', 'input_dir', 'data/input')
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path"""
        return Path(self.config_path.parent) / self.get('paths', 'output_dir', 'data/output')
    
    @property
    def temp_dir(self) -> Path:
        """Get temporary directory path"""
        return Path(self.config_path.parent) / self.get('paths', 'temp_dir', 'data/temp')
    
    @property
    def model_dir(self) -> Path:
        """Get model directory path"""
        return Path(self.config_path.parent) / self.get('paths', 'model_dir', 'models/yolov11')
    
    @property
    def log_dir(self) -> Path:
        """Get log directory path"""
        return Path(self.config_path.parent) / self.get('paths', 'log_dir', 'logs')


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reset_config() -> None:
    """Reset global configuration instance"""
    global _config_instance
    _config_instance = None