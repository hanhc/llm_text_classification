# src/utils/registry.py

import logging
from typing import Dict, Type, Any, Callable

logger = logging.getLogger(__name__)

class Registry:
    """
    A generic registry to map strings to classes or functions.
    This is the core of our factory pattern.
    """
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        """
        A decorator to register a new class or function.

        Args:
            name (str): The name to register the object with.
        """
        def decorator(obj: Any) -> Any:
            if name in self._registry:
                logger.warning(f"{name} is already registered in {self._name}. It will be overwritten.")
            self._registry[name] = obj
            logger.debug(f"Registered {name} in {self._name}")
            return obj
        return decorator

    def get(self, name: str) -> Any:
        """
        Get a registered object by its name.
        """
        if name not in self._registry:
            logger.error(f"Object with name '{name}' not found in {self._name} registry.")
            raise KeyError(f"'{name}' is not a registered name in {self._name}.")
        return self._registry[name]

# Create global registries for different components
DATA_PROCESSOR_REGISTRY = Registry("data_processor")
TUNING_STRATEGY_REGISTRY = Registry("tuning_strategy")