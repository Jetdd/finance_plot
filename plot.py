"""
Author: Jet Deng
Date: 2025-07-28 14:57:36
LastEditTime: 2025-07-28 15:21:35
Description:
"""

from typing import List, Tuple
import pandas as pd
import altair as alt
import warnings

alt.data_transformers.enable("vegafusion")


class FinancePlot:
    def __init__(
        self,
        subplot: bool = False,
        interactive: bool = False,
        standardize: bool = False,
        normalize: bool = False,
    ):
        self.subplot = subplot
        self.interactive = interactive
        self.standardize = standardize
        self.normalize = normalize

    def _standardize(self, series: pd.Series) -> pd.Series:
        """
        Standardize a pandas Series by dividing each element by the first element.

        Args:
            series (pd.Series): The input series to standardize.

        Returns:
            pd.Series: The standardized series.
        """

        return series / series.iloc[0]

    def _normalize(self, series: pd.Series) -> pd.Series:
        return (series - series.mean()) / series.std()

    def preprocess(self, *series, labels: List) -> List:
        # Sanity checks
        for obj in series:
            if not isinstance(obj, pd.Series):
                raise TypeError(f"Expected a pandas series, got {type(obj)} instead.")

        if len(series) < 2 and self.subplot:
            warnings.warn("At least two series are required for subplot=True")
            self.subplot = False

        if self.standardize:
            if len(labels) == len(series):
                dfs = [self._standardize(s).rename(l) for s, l in zip(series, labels)]
            else:
                dfs = [
                    self._standardize(s).rename(f"series_{i + 1}")
                    for i, s in enumerate(series)
                ]
        else:
            if len(labels) == len(series):
                dfs = [s.rename(l) for s, l in zip(series, labels)]
            else:
                dfs = [s.rename(f"series_{i + 1}") for i, s in enumerate(series)]

        if self.normalize:
            if len(labels) == len(series):
                dfs = [self._standardize(s).rename(l) for s, l in zip(dfs, labels)]
            else:
                dfs = [
                    self._standardize(s).rename(f"series_{i + 1}")
                    for i, s in enumerate(dfs)
                ]
        else:
            if len(labels) == len(series):
                dfs = [s.rename(l) for s, l in zip(dfs, labels)]
            else:
                dfs = [s.rename(f"series_{i + 1}") for i, s in enumerate(dfs)]

        return dfs

    def plot_trend(
        self,
        *series,
        labels=[],
        title="Line Plot",
        x_label="Time",
        y_label="Value",
        secondary_y=False,  # False | True | [label/index, ...]
        figsize: Tuple = (600, 300),
    ) -> alt.Chart:
        if self.interactive:
            for s in series:
                if s.shape[0] > 2000:
                    warnings.warn(
                        "Be cautious when using interactive mode for series with more than 2000 elements."
                    )

        dfs = self.preprocess(*series, labels=labels)

        # --- subplot=True：保持原有子图栈叠；不做 secondary_y ---
        if self.subplot:
            if secondary_y:
                warnings.warn(
                    "secondary_y is not supported when subplot=True; fallback to single axis per subplot."
                )
            new_dfs = []
            for s in dfs:
                s.index.name = "time"
                try:
                    s.index = s.index.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    s.index = s.index.strftime("%Y-%m-%d")
                new_dfs.append(
                    s.reset_index().melt("time", var_name="symbol", value_name="value")
                )
            charts = [self._plot_trend(df_) for df_ in new_dfs]
            chart = (
                charts[0]
                .encode(x=alt.X(title=x_label), y=alt.Y(title=y_label))
                .properties(title=title, width=figsize[0], height=figsize[1])
            )
            for c in charts[1:]:
                c = c.encode(x=alt.X(title=x_label), y=alt.Y(title=y_label)).properties(
                    title=title, width=figsize[0], height=figsize[1]
                )
            return chart & c

        # --- 非 subplot：拼成长表 ---
        df = pd.concat(dfs, axis=1).dropna(axis=0, how="all")
        df.index.name = "time"
        try:
            df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            df.index = df.index.strftime("%Y-%m-%d")
        df = df.reset_index().melt("time", var_name="symbol", value_name="value")

        # --- 无 secondary_y：走原有单轴逻辑 ---
        if not secondary_y:
            chart = (
                self._plot_trend(df)
                .encode(
                    x=alt.X(title=x_label),
                    y=alt.Y(title=y_label),
                )
                .properties(title=title, width=figsize[0], height=figsize[1])
            )
            return chart

        # --- secondary_y：分左右轴并叠加 ---
        symbols = list(df["symbol"].unique())
        n_sym = len(symbols)

        if isinstance(secondary_y, bool):
            # True -> 后一半映射到右轴
            right_syms = set(symbols[n_sym // 2 :])
        else:
            if all(isinstance(x, int) for x in secondary_y):
                right_syms = {symbols[i] for i in secondary_y if 0 <= i < n_sym}
            else:
                right_syms = set(secondary_y)

        # 兜底：避免空集或全集都落右轴
        if not right_syms:
            right_syms = {symbols[-1]}
        if len(right_syms) == n_sym:
            right_syms = {symbols[-1]}

        left_df = df[~df["symbol"].isin(right_syms)]
        right_df = df[df["symbol"].isin(right_syms)]

        # —— 左轴主线 ——
        left = (
            alt.Chart(left_df)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:O",
                    title=x_label,
                    axis=alt.Axis(
                        labelAngle=45,
                        labelOverlap="greedy",
                        labelLimit=100,
                        labelPadding=5,
                    ),
                ),
                y=alt.Y(
                    "value:Q", axis=alt.Axis(title=y_label), scale=alt.Scale(zero=False)
                ),
                color=alt.Color("symbol:N").title(None),
            )
        )

        # —— 右轴主线 ——
        right = (
            alt.Chart(right_df)
            .mark_line(strokeDash=[4, 2])
            .encode(
                x=alt.X("time:O", title=x_label),
                y=alt.Y(
                    "value:Q",
                    axis=alt.Axis(title="Secondary Y", orient="right"),
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("symbol:N").title(None),
                opacity=alt.value(0.9),
            )
        )

        if self.interactive:
            # 共享 hover（基于 time）
            nearest = alt.selection_point(
                nearest=True, on="pointerover", fields=["time"], empty=False
            )

            # 左侧高亮点（放在左层内部）
            left_pts = (
                alt.Chart(left_df)
                .mark_point()
                .encode(
                    x="time:O",
                    y="value:Q",
                    color="symbol:N",
                    opacity=alt.when(nearest)
                    .then(alt.value(1))
                    .otherwise(alt.value(0)),
                )
                .transform_filter(nearest)
            )
            left_layer = alt.layer(left, left_pts)

            # 右侧高亮点（放在右层内部）
            right_pts = (
                alt.Chart(right_df)
                .mark_point()
                .encode(
                    x="time:O",
                    y="value:Q",
                    color="symbol:N",
                    opacity=alt.when(nearest)
                    .then(alt.value(1))
                    .otherwise(alt.value(0)),
                )
                .transform_filter(nearest)
            )
            right_layer = alt.layer(right, right_pts)

            # 主图：左右层并列，并在此处声明独立 y 轴（避免“串台”）
            chart = (
                alt.layer(left_layer, right_layer)
                .resolve_scale(y="independent")
                .properties(title=title, width=figsize[0], height=figsize[1])
            )

            # 隐形选择器（无 y 编码）
            selectors = (
                alt.Chart(df)
                .mark_point()
                .encode(x="time:O", opacity=alt.value(0))
                .add_params(nearest)
            )

            # 宽表 tooltip（同一时刻展示所有 symbol 的值；无 y 编码）
            tooltip_cols = [alt.Tooltip("time:O", title="Time")] + [
                alt.Tooltip(f"{s}:Q", title=s) for s in symbols
            ]
            tool = (
                alt.Chart(df)
                .transform_pivot("symbol", value="value", groupby=["time"])
                .mark_rule(color="gray")
                .encode(x="time:O", tooltip=tooltip_cols)
                .transform_filter(nearest)
            )

            return chart + selectors + tool

        # 非交互：简单左右叠加 + 独立 y
        return (
            alt.layer(left, right)
            .resolve_scale(y="independent")
            .properties(title=title, width=figsize[0], height=figsize[1])
        )

    def _plot_trend(self, df: pd.DataFrame) -> alt.Chart:
        if self.interactive:
            nearest = alt.selection_point(
                nearest=True, on="pointerover", fields=["time"], empty=False
            )

            line = (
                alt.Chart(df)
                .mark_line(interpolate="basis")
                .encode(
                    x=alt.X(
                        "time:O",
                        axis=alt.Axis(
                            labelAngle=45,
                            labelOverlap="greedy",
                            labelLimit=100,
                            labelPadding=5,
                        ),
                    ),
                    y=alt.Y("value:Q").scale(zero=False),
                    color=alt.Color(
                        "symbol:N",
                        legend=alt.Legend(
                            title=None,
                            orient="right",
                            direction="vertical",
                            symbolType="stroke",
                        ),
                    ),
                )
            )

            selectors = (
                alt.Chart(df)
                .mark_point()
                .encode(
                    x="time:O",
                    opacity=alt.value(0),
                )
                .add_params(nearest)
            )
            when_near = alt.when(nearest)
            points = line.mark_point().encode(
                opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
            )

            rules = (
                alt.Chart(df)
                .mark_rule(color="gray")
                .encode(
                    x="time:O",
                    tooltip=[
                        alt.Tooltip("time:O", title="Time"),
                        # 注意：这里列名要和 pivot 后的一致
                        *[
                            alt.Tooltip(sym + ":Q", title=sym)
                            for sym in df["symbol"].unique()
                        ],
                    ],
                )
                .transform_filter(nearest)
                # pivot 变成宽表，每个 symbol 一列
                .transform_pivot("symbol", value="value", groupby=["time"])
            )

            chart = line + selectors + points + rules

        else:
            base = alt.Chart(df).encode(
                alt.Color(
                    "symbol:N",
                    legend=alt.Legend(
                        title=None,
                        orient="right",
                        direction="vertical",
                        symbolType="stroke",
                    ),
                )
            )

            line = base.mark_line().encode(
                x=alt.X(
                    "time:O",
                    axis=alt.Axis(
                        labelAngle=45,
                        labelOverlap="greedy",
                        labelLimit=100,
                        labelPadding=5,
                    ),
                ),
                y=alt.Y("value:Q").scale(zero=False),
            )

            last_loc = (
                base.mark_circle()
                .encode(
                    alt.X("last_time['time']:O"),
                    alt.Y("last_time['value']:Q"),
                )
                .transform_aggregate(
                    last_time="argmax(time)",
                    groupby=["symbol"],
                )
            )

            symbols = last_loc.mark_text(align="left", dx=4).encode(text="symbol")

            chart = line + last_loc + symbols

        return chart

    def plot_distribution(
        self, *series, bins: int = 20, labels: list = []
    ) -> alt.Chart:
        dfs = self.preprocess(*series, labels=labels)

        if self.subplot:
            new_dfs = []
            for series in dfs:
                series.index.name = "time"
                new_dfs.append(
                    series.reset_index().melt(
                        "time", var_name="symbol", value_name="value"
                    )
                )

            charts = [self._plot_distribution(df, bins) for df in new_dfs]

            chart = charts[0].properties(
                width=600, height=300, title=f"Distribution for {dfs[0].name}"
            )
            for i, c in enumerate(charts[1:]):
                c = c.properties(
                    width=600, height=300, title=f"Distribution for {dfs[i + 1].name}"
                )
            chart = chart & c

        else:
            warnings.warn("Distribution plot is not supported for subplot=False")

        return chart

    def _plot_distribution(self, df: pd.DataFrame, bins: int) -> alt.Chart:
        if self.interactive:
            select = alt.selection_point(on="click")
            highlight = alt.selection_point(on="pointerover", empty=False)

            stroke_width = (
                alt.when(select)
                .then(alt.value(2, empty=False))
                .when(highlight)
                .then(alt.value(1))
                .otherwise(alt.value(0))
            )

            chart = (
                alt.Chart(df, height=200)
                .mark_bar(
                    stroke="black",
                    cursor="pointer",
                )
                .encode(
                    alt.X("value:Q", bin=alt.Bin(maxbins=bins)),
                    y="count()",
                    strokeWidth=stroke_width,
                    fillOpacity=alt.when(select)
                    .then(alt.value(1))
                    .otherwise(alt.value(0.3)),
                )
                .add_params(select, highlight)
            )
        else:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    alt.X("value:Q", bin=alt.Bin(maxbins=bins)),
                    y="count()",
                )
            )

        return chart

    def plot_corr(self, *series, labels: list = []) -> alt.Chart:
        if self.standardize:
            if len(labels) == len(series):
                dfs = [self._standardize(s).rename(l) for s, l in zip(series, labels)]
            else:
                dfs = [
                    self._standardize(s).rename(f"series_{i + 1}")
                    for i, s in enumerate(series)
                ]
        else:
            if len(labels) == len(series):
                dfs = [s.rename(l) for s, l in zip(series, labels)]
            else:
                dfs = [s.rename(f"series_{i + 1}") for i, s in enumerate(series)]

        if self.subplot:
            warnings.warn("Correlation plot is not supported for subplot=True")
            return

        df = pd.concat(dfs, axis=1)
        chart = self._plot_corr(df)
        return chart

    def _plot_corr(self, df: pd.DataFrame) -> alt.Chart:
        df.index.name = "time"
        try:
            df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
        except:
            df.index = df.index.strftime("%Y-%m-%d")
        df = df.reset_index()

        if self.interactive:
            select_x = alt.selection_point(
                fields=["level_0"],
                name="select_x",
                value="b",
            )
            select_y = alt.selection_point(
                fields=["level_1"],
                name="select_y",
                value="d",
            )

            heatmap = (
                alt.Chart(
                    df.drop(columns="time")
                    .corr()
                    .stack()
                    .reset_index()
                    .rename(columns={0: "correlation"}),
                    title="Correlation",
                    height=250,
                    width=250,
                )
                .mark_rect()
                .encode(
                    x=alt.X("level_0").title(None),
                    y=alt.Y("level_1").title(None),
                    color=alt.Color("correlation").scale(
                        domain=[-1, 1], scheme="blueorange"
                    ),
                    opacity=alt.when(select_x, select_y)
                    .then(alt.value(1))
                    .otherwise(alt.value(0.4)),
                )
                .add_params(select_x, select_y)
            )

            base = alt.Chart(
                df.melt("time", var_name="symbol", value_name="value"),
                height=100,
                width=300,
                title="Time Series",
            )

            lines = (
                base.transform_filter(
                    "indexof(datum.symbol, select_x.level_0) !== -1"
                    "| indexof(datum.symbol, select_y.level_1) !== -1"
                )
                .mark_line()
                .encode(
                    alt.X(
                        "time:O",
                        axis=alt.Axis(
                            labelAngle=45,
                            labelOverlap="greedy",
                            labelLimit=100,
                            labelPadding=5,
                        ),
                    ).title(None),
                    alt.Y("value:Q").scale(zero=False),
                    alt.Color("symbol:N").legend(orient="top", offset=-20).title(None),
                )
            )

            dynamic_title = alt.Title(
                alt.expr(
                    f'"Difference " + {select_x.name}.level_0 + " - " + {select_y.name}.level_1'
                )
            )

            lines_diff = (
                base.transform_pivot(
                    "symbol",
                    "value",
                    groupby=["time"],
                    # In the calculate transform we use the values from the selection to subset the columns to substract
                )
                .transform_calculate(
                    difference=f"datum[{select_x.name}.level_0] - datum[{select_y.name}.level_1]"
                )
                .mark_line(color="grey")
                .encode(
                    alt.X(
                        "time:O",
                        axis=alt.Axis(
                            labelAngle=45,
                            labelOverlap="greedy",
                            labelLimit=100,
                            labelPadding=5,
                        ),
                    ).title(None),
                    alt.Y("difference:Q").scale(zero=False),
                )
                .properties(title=dynamic_title)
            )

            chart = (lines & lines_diff) | heatmap

        else:
            df = df.drop(columns="time").corr()
            chart = (
                alt.Chart(
                    df.stack().reset_index().rename(columns={0: "correlation"}),
                    title="Correlation Plot",
                    height=250,
                    width=250,
                )
                .mark_rect()
                .encode(
                    x=alt.X("level_0").title(None),
                    y=alt.Y("level_1").title(None),
                    color=alt.Color("correlation").scale(
                        domain=[-1, 1], scheme="redblue"
                    ),
                )
            )

        return chart
