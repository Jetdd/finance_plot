"""
Seaborn/Matplotlib implementation of your plot.py API.

Changes added per request:
1) Explicit different colors for different lines (stable symbol->color map)
2) Remove time gaps in trend plots (plot on continuous index t_idx, show time as tick labels)
3) Add grid on trend plots (research style; for secondary_y apply grid only on left axis)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Union, Optional, Dict, Set

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


SecondaryYSpec = Union[bool, Sequence[Union[int, str]]]


@dataclass
class FinancePlotSeaborn:
    subplot: bool = False
    interactive: bool = (
        False  # kept for API compatibility; no Altair-like interaction in seaborn
    )
    standardize: bool = False
    normalize: bool = False

    # palette: None -> use matplotlib TABLEAU_COLORS
    palette: Optional[Sequence[str]] = None

    # show time ticks on x-axis for no-gap plotting
    show_time_ticks: bool = True
    max_xticks: int = 6  # max number of x ticks to show per axis

    # grid style
    grid: bool = True
    grid_minor: bool = False  # True -> enable minor grid (heavier look)

    # -----------------------------
    # Preprocess (same semantics)
    # -----------------------------
    def _standardize(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return series
        first = series.iloc[0]
        if pd.isna(first) or first == 0:
            return series
        return series / first

    def _normalize(self, series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std == 0 or pd.isna(std):
            return series * 0.0
        return (series - series.mean()) / std

    def preprocess(self, *series: pd.Series, labels: List) -> List[pd.Series]:
        for obj in series:
            if not isinstance(obj, pd.Series):
                raise TypeError(f"Expected pandas.Series, got {type(obj)}")

        if len(series) < 2 and self.subplot:
            warnings.warn(
                "At least two series are required for subplot=True; fallback to subplot=False"
            )
            self.subplot = False

        # assign names
        if len(labels) == len(series):
            named = [s.rename(l) for s, l in zip(series, labels)]
        else:
            named = [s.rename(f"series_{i + 1}") for i, s in enumerate(series)]

        # apply transforms
        if self.standardize:
            named = [self._standardize(s).rename(s.name) for s in named]
        if self.normalize:
            named = [self._normalize(s).rename(s.name) for s in named]

        return named

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _to_wide_df(dfs: List[pd.Series]) -> pd.DataFrame:
        df = pd.concat(dfs, axis=1)
        df.index.name = "time"
        return df

    @staticmethod
    def _to_long_df(dfs: List[pd.Series]) -> pd.DataFrame:
        df = FinancePlotSeaborn._to_wide_df(dfs).dropna(axis=0, how="all")
        long_df = df.reset_index().melt("time", var_name="symbol", value_name="value")
        return long_df

    def _resolve_palette(self) -> List[str]:
        if self.palette is not None and len(self.palette) > 0:
            return list(self.palette)
        return list(mcolors.TABLEAU_COLORS.values())

    def _resolve_colors(self, symbols: List[str]) -> Dict[str, str]:
        pal = self._resolve_palette()
        # stable mapping (deterministic)
        colors = {}
        for i, sym in enumerate(symbols):
            colors[sym] = pal[i % len(pal)]
        return colors

    @staticmethod
    def _resolve_right_syms(
        symbols: List[str], secondary_y: SecondaryYSpec
    ) -> Set[str]:
        n = len(symbols)
        if isinstance(secondary_y, bool):
            # True -> second half
            right = set(symbols[n // 2 :])
        else:
            # explicit list (indices or labels)
            if all(isinstance(x, int) for x in secondary_y):
                right = {symbols[i] for i in secondary_y if 0 <= i < n}
            else:
                right = {str(x) for x in secondary_y}

        # guardrails: don't allow empty or all
        if not right and symbols:
            right = {symbols[-1]}
        if len(right) == n and symbols:
            right = {symbols[-1]}
        return right

    def _apply_grid(self, ax: plt.Axes) -> None:
        if not self.grid:
            return
        ax.set_axisbelow(True)
        ax.grid(
            True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.4
        )
        if self.grid_minor:
            ax.minorticks_on()
            ax.grid(
                True,
                which="minor",
                axis="both",
                linestyle=":",
                linewidth=0.5,
                alpha=0.2,
            )

    def _apply_time_ticks(self, ax: plt.Axes, df_any_sym: pd.DataFrame) -> None:
        """
        df_any_sym: must contain columns ["t_idx","time"] and already sorted by t_idx.
        """
        if not self.show_time_ticks or df_any_sym.empty:
            return

        n = len(df_any_sym)
        if n <= 1:
            return

        step = max(n // self.max_xticks, 1)
        tick_idx = df_any_sym["t_idx"].iloc[::step].to_numpy()
        tick_lbl = df_any_sym["time"].iloc[::step].astype(str).to_numpy()

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_lbl, rotation=45, ha="right")

    # -----------------------------
    # plot_trend (no-gap + colors + grid)
    # -----------------------------
    def plot_trend(
        self,
        *series: pd.Series,
        labels: List[str] = [],
        title: str = "Line Plot",
        x_label: str = "Time",
        y_label: str = "Value",
        secondary_y: SecondaryYSpec = False,
        secondary_y_label: Optional[str] = "Secondary Y",
        figsize: Tuple[int, int] = (10, 4),
    ):
        """
        seaborn/mpl trend plot with:
        - explicit color map
        - no time gap: use t_idx for x, show time as ticks
        - grid enabled by default

        Returns:
          - subplot=False & secondary_y=False: (fig, ax)
          - subplot=False & secondary_y!=False: (fig, (ax, ax2))
          - subplot=True: (fig, axes)
        """
        if self.interactive:
            for s in series:
                if getattr(s, "shape", (0,))[0] > 2000:
                    warnings.warn(
                        "interactive=True has no effect in seaborn/matplotlib."
                    )

        dfs = self.preprocess(*series, labels=labels)

        # -------- subplot mode --------
        if self.subplot:
            if secondary_y:
                warnings.warn(
                    "secondary_y is not supported when subplot=True; ignoring secondary_y."
                )
            n = len(dfs)
            fig, axes = plt.subplots(
                n, 1, figsize=(figsize[0], figsize[1] * n), sharex=False
            )
            if n == 1:
                axes = [axes]

            for ax, s in zip(axes, dfs):
                df_ = s.dropna().reset_index()
                df_.columns = ["time", "value"]
                df_ = df_.sort_values("time")
                df_["t_idx"] = np.arange(len(df_))

                color_map = self._resolve_colors([s.name])
                ax.plot(
                    df_["t_idx"],
                    df_["value"],
                    color=color_map[s.name],
                    linewidth=1.8,
                    label=s.name,
                )

                ax.set_title(f"{title} — {s.name}")
                ax.set_ylabel(y_label)
                ax.set_xlabel(x_label)
                self._apply_grid(ax)
                self._apply_time_ticks(ax, df_)

                ax.legend(title=None, loc="best")

            fig.tight_layout()
            return fig, axes

        # -------- non-subplot --------
        long_df = self._to_long_df(dfs)
        if long_df.empty:
            raise ValueError("No data to plot after preprocessing/dropna.")

        # sort by time and create no-gap x per symbol
        long_df = long_df.sort_values(["time", "symbol"])
        long_df["t_idx"] = long_df.groupby("symbol").cumcount()

        symbols = [s.name for s in dfs]
        color_map = self._resolve_colors(symbols)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if not secondary_y:
            # plot each symbol manually for full color control
            for sym in symbols:
                sub = long_df[long_df["symbol"] == sym]
                ax.plot(
                    sub["t_idx"],
                    sub["value"],
                    label=sym,
                    color=color_map[sym],
                    linewidth=1.8,
                )

            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            self._apply_grid(ax)

            # use the first symbol's time for ticks (all share same "t_idx" meaning: order count)
            any_sym = symbols[0]
            sub0 = long_df[long_df["symbol"] == any_sym].sort_values("t_idx")
            self._apply_time_ticks(ax, sub0)

            ax.legend(title=None, loc="best")
            fig.tight_layout()
            return fig, ax

        # secondary y: split symbols
        right_syms = self._resolve_right_syms(symbols, secondary_y)
        left_syms = [s for s in symbols if s not in right_syms]
        right_syms = list(right_syms)

        # left axis
        for sym in left_syms:
            sub = long_df[long_df["symbol"] == sym]
            ax.plot(
                sub["t_idx"],
                sub["value"],
                label=sym,
                color=color_map[sym],
                linewidth=1.8,
            )
        ax.set_ylabel(y_label)

        # right axis
        ax2 = ax.twinx()
        for sym in right_syms:
            sub = long_df[long_df["symbol"] == sym]
            ax2.plot(
                sub["t_idx"],
                sub["value"],
                label=sym,
                color=color_map[sym],
                linestyle="--",
                linewidth=1.8,
            )
        ax2.set_ylabel(secondary_y_label)

        ax.set_title(title)
        ax.set_xlabel(x_label)

        # grid: apply ONLY on left axis to avoid double grid overlay
        self._apply_grid(ax)

        # ticks from the first available series (prefer left if exists)
        tick_sym = left_syms[0] if left_syms else right_syms[0]
        sub_tick = long_df[long_df["symbol"] == tick_sym].sort_values("t_idx")
        self._apply_time_ticks(ax, sub_tick)

        # merge legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, title=None, loc="best")
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()

        fig.tight_layout()
        return fig, (ax, ax2)

    # -----------------------------
    # plot_distribution (kept simple; explicit overlay)
    # -----------------------------
    def plot_distribution(
        self,
        *series: pd.Series,
        bins: int = 20,
        labels: List[str] = [],
        kde: bool = False,
        stat: str = "count",  # "count" | "density"
        figsize: Tuple[int, int] = (10, 4),
    ):
        dfs = self.preprocess(*series, labels=labels)
        if not dfs:
            raise ValueError("plot_distribution requires at least one series.")

        # combined frame
        frames = []
        for s in dfs:
            tmp = pd.DataFrame({"value": s.dropna().values, "series_label": s.name})
            frames.append(tmp)
        combined = pd.concat(frames, ignore_index=True)

        if self.subplot and len(dfs) > 1:
            n = len(dfs)
            fig, axes = plt.subplots(
                n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True
            )
            if n == 1:
                axes = [axes]

            for ax, s in zip(axes, dfs):
                sns.histplot(s.dropna().values, bins=bins, kde=kde, stat=stat, ax=ax)
                ax.set_title(f"Distribution — {s.name}")
                ax.set_ylabel(stat)
            axes[-1].set_xlabel("value")
            fig.tight_layout()
            return fig, axes

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.histplot(
            data=combined,
            x="value",
            hue="series_label",
            bins=bins,
            kde=kde,
            stat=stat,
            element="step",
            common_norm=(stat == "density"),
            ax=ax,
        )
        ax.set_title("Distribution")
        ax.set_xlabel("value")
        ax.set_ylabel(stat)
        ax.legend(title=None, loc="best")
        fig.tight_layout()
        return fig, ax

    # -----------------------------
    # plot_corr (heatmap)
    # -----------------------------
    def plot_corr(
        self,
        *series: pd.Series,
        labels: List[str] = [],
        method: str = "pearson",
        annot: bool = False,
        fmt: str = ".2f",
        figsize: Tuple[int, int] = (6, 5),
        vmin: float = -1.0,
        vmax: float = 1.0,
        cmap: str = "RdBu_r",
    ):
        if self.subplot:
            warnings.warn("Correlation plot is not supported for subplot=True")
            return None

        dfs = self.preprocess(*series, labels=labels)
        df = pd.concat(dfs, axis=1)
        corr = df.corr(method=method)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(
            corr,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            square=True,
            ax=ax,
        )
        ax.set_title("Correlation Plot")
        fig.tight_layout()
        return fig, ax
