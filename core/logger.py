"""
Logging system for Nong-View2
"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional
from .config import get_config


class LoggerSetup:
    """Logger configuration and setup"""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize logger with configuration
        
        Args:
            config: Logger configuration dictionary
        """
        self.config = config or get_config().get('logging', {})
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with configuration"""
        # Remove default handler
        logger.remove()
        
        # Get configuration values
        level = self.config.get('level', 'INFO')
        format_str = self.config.get('format', '{time} | {level} | {message}')
        rotation = self.config.get('rotation', '100 MB')
        retention = self.config.get('retention', '7 days')
        colorize = self.config.get('colorize', True)
        
        # Add console handler
        logger.add(
            sys.stdout,
            format=format_str,
            level=level,
            colorize=colorize
        )
        
        # Add file handler
        log_dir = get_config().log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_dir / "nongview_{time}.log",
            format=format_str,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding='utf-8'
        )
    
    @staticmethod
    def get_logger(name: str = None):
        """Get logger instance
        
        Args:
            name: Logger name (module name)
            
        Returns:
            Logger instance
        """
        if name:
            return logger.bind(name=name)
        return logger


# Initialize global logger
_logger_setup = None


def setup_logger(config: Optional[dict] = None):
    """Setup global logger
    
    Args:
        config: Logger configuration
    """
    global _logger_setup
    if _logger_setup is None:
        _logger_setup = LoggerSetup(config)
    return _logger_setup


def get_logger(name: str = None):
    """Get logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if _logger_setup is None:
        setup_logger()
    return LoggerSetup.get_logger(name)


# Initialize logger on module import
setup_logger()