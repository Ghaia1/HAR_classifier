from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_recording_interactive(
    df: pd.DataFrame,
    output_path: Path,
    crop_start_s: float | None = None,
    crop_end_s: float | None = None,
) -> None:
    """Create an interactive HTML plot of one recording with subplots for ACC and GYR.
    
    Args:
        df: Recording DataFrame with 'time_s', 'acc_*', 'gyr_*' columns
        output_path: Where to save the HTML
        crop_start_s: Optional time (s) to mark as crop start with a vertical line
        crop_end_s: Optional time (s) to mark as crop end with a vertical line
    """
    
    rec_name = df["recording"].iloc[0]
    activity = df["activity_name"].iloc[0]
    samples = len(df)
    
    # Count samples in cropped region
    cropped_samples = samples
    if crop_start_s is not None and crop_end_s is not None:
        cropped_df = df[(df["time_s"] >= crop_start_s) & (df["time_s"] <= crop_end_s)]
        cropped_samples = len(cropped_df)
    
    title_suffix = f" | {samples} raw samples, {cropped_samples} after crop" if (crop_start_s is not None) else f" | {samples} samples"
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Accelerometer — {rec_name} ({activity}){title_suffix}",
            "Gyroscope"
        ),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )
    
    # Accelerometer traces
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["acc_x"], mode="lines", name="acc_x",
                   line=dict(width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["acc_y"], mode="lines", name="acc_y",
                   line=dict(width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["acc_z"], mode="lines", name="acc_z",
                   line=dict(width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["acc_norm"], mode="lines", name="acc_norm",
                   line=dict(width=2, dash="solid")),
        row=1, col=1
    )
    
    # Gyroscope traces
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["gyr_x"], mode="lines", name="gyr_x",
                   line=dict(width=1), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["gyr_y"], mode="lines", name="gyr_y",
                   line=dict(width=1), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["gyr_z"], mode="lines", name="gyr_z",
                   line=dict(width=1), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["gyr_norm"], mode="lines", name="gyr_norm",
                   line=dict(width=2, dash="solid"), showlegend=False),
        row=2, col=1
    )
    
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Gyroscope (rad/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    
    # Add crop boundary lines
    if crop_start_s is not None:
        fig.add_vline(x=crop_start_s, line_dash="dash", line_color="red", opacity=0.6,
                      annotation_text="crop_start", annotation_position="top left")
    
    if crop_end_s is not None:
        fig.add_vline(x=crop_end_s, line_dash="dash", line_color="orange", opacity=0.6,
                      annotation_text="crop_end", annotation_position="top right")
    
    fig.update_layout(
        height=800,
        width=1400,
        hovermode="x unified",
        template="plotly_white",
    )
    
    fig.write_html(str(output_path))
