import pandas as pd
import ta
import plotly.graph_objects as go

from .indicator import Indicator


class StochasticIndicator(Indicator):
    """
    Stochastic Oscillator (%K / %D) — TradingView-style.

    Supports four variants via `suffix` so columns don't collide:
      suffix=""  -> stoch_k,  stoch_d
      suffix="2" -> stoch_k2, stoch_d2
      suffix="3" -> stoch_k3, stoch_d3
      suffix="4" -> stoch_k4, stoch_d4
    """

    def __init__(
        self,
        k_window,
        k_smoothing,
        d_smoothing,
        *,
        suffix: str = "",       # "", "2", "3", or "4"
    ):
        title = "Stochastic" + (suffix if suffix else "")
        super().__init__(title)

        self.k_window = k_window
        self.k_smoothing = k_smoothing
        self.d_smoothing = d_smoothing
        self.suffix = suffix

        # DataFrame column names for this variant
        prefix = "stoch"
        self.k_col = f"{prefix}_k{self.suffix}"
        self.d_col = f"{prefix}_d{self.suffix}"

    # ------------------------------------------------------------------ #
    # Calculation
    # ------------------------------------------------------------------ #
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append this variant's %K/%D columns to *df*.
        Requires columns: `high`, `low`, `close`.
        """
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.k_window,           # %K length
            smooth_window=self.k_smoothing  # %K smoothing
        )

        # %K after K-smoothing (TradingView-style %K)
        df[self.k_col] = stoch.stoch()

        # %D = SMA(%K, d_smoothing)
        df[self.d_col] = (
            df[self.k_col]
            .rolling(window=self.d_smoothing, min_periods=self.d_smoothing)
            .mean()
        )

        return df

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """
        Add this variant's traces to *fig* (uses df['timestamp']).
        Always shows %K, %D, and 20/80 guides, with y-axis locked to [0, 100].
        """
        x = df["timestamp"]

        # # %K line (always shown)
        # fig.add_trace(
        #     go.Scatter(
        #         x=x,
        #         y=df[self.k_col],
        #         mode="lines",
        #         line=dict(width=1),
        #         name=f"%K{self.suffix} ({self.k_window},{self.k_smoothing})",
        #         hovertemplate="%K: %{y:.1f}<extra></extra>",
        #         connectgaps=True,
        #     ),
        #     row=row, col=1
        # )

        # %D line (always shown)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[self.d_col],
                mode="lines",
                line=dict(width=1),
                name=f"%D{self.suffix} ({self.d_smoothing})",
                hovertemplate="%D: %{y:.1f}<extra></extra>",
                connectgaps=True,
            ),
            row=row, col=1
        )

        # 20/80 guides (always added)
        n = len(df)
        fig.add_trace(
            go.Scatter(
                x=x, y=[80] * n,
                mode="lines",
                line=dict(width=1, color="white"),
                opacity=0.6,
                name="Overbought (80)",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=[20] * n,
                mode="lines",
                line=dict(width=1, color="white"),
                opacity=0.6,
                name="Oversold (20)",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=1
        )

        # Keep axis in [0,100] (locked)
        fig.update_yaxes(
            title_text=f"Stoch{self.suffix} %",
            range=[0, 100],
            fixedrange=True,
            row=row, col=1
        )
