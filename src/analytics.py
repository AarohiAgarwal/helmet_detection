"""
Analytics and Visualizations Engine for Helmet Detection Dashboard.
Generates responsive Plotly charts for compliance ratios, trends, and class breakdowns.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class AnalyticsEngine:
    """Generates Plotly interactive charts from genuine detection records."""

    @staticmethod
    def get_helmet_vs_nohelmet_bar(metrics: Dict[str, Any]) -> go.Figure:
        """Creates a modern comparison bar chart between Helmet and NoHelmet counts."""
        helmets = metrics.get("total_helmets", 0)
        no_helmets = metrics.get("total_no_helmets", 0)

        fig = go.Figure(data=[
            go.Bar(
                x=["Helmet (Compliant)", "NoHelmet (Violation)"],
                y=[helmets, no_helmets],
                marker_color=["#00CC96", "#EF553B"],
                text=[f"{helmets} ({metrics.get('helmet_pct', 0):.1f}%)", f"{no_helmets} ({metrics.get('no_helmet_pct', 0):.1f}%)"],
                textposition="auto"
            )
        ])

        fig.update_layout(
            title="Helmet Compliance vs. Violation Counts",
            xaxis_title="Category",
            yaxis_title="Count",
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @staticmethod
    def get_detection_distribution_pie(df: pd.DataFrame, metrics: Dict[str, Any]) -> go.Figure:
        """Creates a donut / pie chart showing overall detection distribution."""
        if not df.empty and "class_name" in df.columns:
            counts = df["class_name"].value_counts().reset_index()
            counts.columns = ["Class", "Count"]
        else:
            counts = pd.DataFrame({
                "Class": ["BikeWithRider", "NoHelmet", "Helmet"],
                "Count": [metrics.get("total_motorcycles", 0), metrics.get("total_no_helmets", 0), metrics.get("total_helmets", 0)]
            })

        colors = {
            "BikeWithRider": "#1f77b4",
            "NoHelmet": "#d62728",
            "Helmet": "#2ca02c"
        }
        color_list = [colors.get(c, "#636EFA") for c in counts["Class"]]

        fig = px.pie(
            counts,
            values="Count",
            names="Class",
            title="Object Detection Distribution",
            color="Class",
            color_discrete_map=colors,
            hole=0.45,
            template="plotly_dark"
        )
        fig.update_layout(height=380, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @staticmethod
    def get_daily_violation_trend(df: pd.DataFrame) -> go.Figure:
        """Plots the daily or hourly trend of detected violations."""
        if df.empty or "timestamp" not in df.columns:
            # Empty placeholder chart
            fig = go.Figure()
            fig.update_layout(
                title="Violation Incident Trend (No data recorded yet)",
                template="plotly_dark",
                height=380
            )
            return fig

        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
        trend = df_copy.groupby("date").size().reset_index(name="Incident Count")

        fig = px.line(
            trend,
            x="date",
            y="Incident Count",
            title="Daily Traffic Safety Incidents & Violations",
            markers=True,
            template="plotly_dark",
            color_discrete_sequence=["#EF553B"]
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Detected Incidents",
            height=380,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @staticmethod
    def get_confidence_distribution(df: pd.DataFrame) -> go.Figure:
        """Displays confidence score distribution histogram across detected classes."""
        if df.empty or "confidence" not in df.columns:
            fig = go.Figure()
            fig.update_layout(title="Confidence Distribution (Awaiting Detections)", template="plotly_dark", height=380)
            return fig

        fig = px.histogram(
            df,
            x="confidence",
            color="class_name",
            nbins=20,
            title="Confidence Score Distribution by Class",
            template="plotly_dark",
            barmode="overlay",
            color_discrete_map={
                "BikeWithRider": "#1f77b4",
                "NoHelmet": "#d62728",
                "Helmet": "#2ca02c"
            }
        )
        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            height=380,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

