"""Automated decline curve analysis (Auto-DCA) for oil and gas production forecasting."""

from .analysis import DCAAnalyzer
from .data_loader import DCADataLoader
from .models import DeclineModels
from .utils import DataPreprocessor, MetricsCalculator
from .visualization import DCAVisualizer

__version__ = "1.0.0"

__all__ = [
    "DCAAnalyzer",
    "DCADataLoader",
    "DeclineModels",
    "DataPreprocessor",
    "MetricsCalculator",
    "DCAVisualizer",
    "__version__",
]
