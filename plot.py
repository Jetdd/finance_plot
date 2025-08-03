"""
Author: Jet Deng
Date: 2025-07-28 14:57:36
LastEditTime: 2025-07-28 15:21:35
Description:
"""

from typing import List
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
                raise ValueError("Only pd.Series accepted")

        if self.subplot == True and len(series) < 2:
            warnings.warn("Subplot should be False when there is only one series")
            self.subplot = False
        
        if self.standardize:
            if len(labels) == len(series):
                dfs = [self._standardize(s).rename(l) for s, l in zip(series, labels)]
            else:
                dfs = [
                    self._standardize(s).rename(f"series_{i+1}")
                    for i, s in enumerate(series)
                ]
        else:
            if len(labels) == len(series):
                dfs = [s.rename(l) for s, l in zip(series, labels)]
            else:
                dfs = [s.rename(f"series_{i+1}") for i, s in enumerate(series)]

        if self.normalize:
            if len(labels) == len(series):
                dfs = [self._standardize(s).rename(l) for s, l in zip(dfs, labels)]
            else:
                dfs = [
                    self._standardize(s).rename(f"series_{i+1}")
                    for i, s in enumerate(dfs)
                ]
        else:
            if len(labels) == len(series):
                dfs = [s.rename(l) for s, l in zip(dfs, labels)]
            else:
                dfs = [s.rename(f"series_{i+1}") for i, s in enumerate(dfs)]

        return dfs

    def plot_trend(
        self, *series, labels=[], title="Line Plot", x_label="Time", y_label="Value"
    ) -> alt.Chart:
        if self.interactive:
            for s in series:
                if s.shape[0] > 2000:
                    warnings.warn(
                        "Be cautious when using interactive mode for series with more than 2000 elements."
                    )

        dfs = self.preprocess(*series, labels=labels)

        if not self.subplot:
            df = pd.concat(dfs, axis=1).dropna(axis=0, how="all")
            df.index.name = "time"

            try:
                df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
            except:
                df.index = df.index.strftime("%Y-%m-%d")

            df = df.reset_index().melt("time", var_name="symbol", value_name="value")

            chart = self._plot_trend(df)
            chart = chart.encode(
                x=alt.X(title=x_label),
                y=alt.Y(title=y_label),
            ).properties(title=title)

        else:
            new_dfs = []
            for series in dfs:
                series.index.name = "time"
                try:
                    series.index = series.index.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    series.index = series.index.strftime("%Y-%m-%d")

                new_dfs.append(
                    series.reset_index().melt(
                        "time", var_name="symbol", value_name="value"
                    )
                )

            charts = [self._plot_trend(df) for df in new_dfs]

            chart = (
                charts[0]
                .encode(
                    x=alt.X(title=x_label),
                    y=alt.Y(title=y_label),
                )
                .properties(title=title)
            )
            for c in charts[1:]:
                c = c.encode(
                    x=alt.X(title=x_label),
                    y=alt.Y(title=y_label),
                ).properties(title=title)
            chart = chart & c

        return chart

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
                    color=alt.Color("symbol:N").legend(None),
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
                .mark_rule(
                    color="gray",
                )
                .encode(
                    x="time:O",
                    tooltip=alt.Tooltip(["time", "value", "symbol"]),
                )
                .transform_filter(nearest)
            )

            chart = line + selectors + points + rules

        else:
            base = alt.Chart(df).encode(alt.Color("symbol").legend(None))

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

        chart = chart.properties(width=600, height=300)

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
                    width=600, height=300, title=f"Distribution for {dfs[i+1].name}"
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
                    self._standardize(s).rename(f"series_{i+1}")
                    for i, s in enumerate(series)
                ]
        else:
            if len(labels) == len(series):
                dfs = [s.rename(l) for s, l in zip(series, labels)]
            else:
                dfs = [s.rename(f"series_{i+1}") for i, s in enumerate(series)]

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
    
    