import pandas as pd
import plotly.graph_objects as go

from abc import ABC, abstractmethod

class Indicator(ABC):
    """
    Abstract base class for technical indicators.
    """

    def __init__(self, name: str):
        self.name = name
    
    # TODO: instead of passing around DataFrame, there is probably a better approach later
    # TODO: we should handle parameters dynamically so we don't have to manually call each separate indicator class
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values and add columns to DataFrame."""
        pass

    @abstractmethod
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add this indicator's traces to the figure."""
        pass