"""
Agent基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logging.basicConfig(level=logging.INFO)


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        处理数据的主方法
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理后的数据
        """
        pass
    
    def log(self, message: str, level: str = "info"):
        """记录日志"""
        log_func = getattr(self.logger, level.lower())
        log_func(f"[{self.name}] {message}")
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        return input_data is not None
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"

