import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Use the provided file path for your CSV. Ensure this path is correct
CSV_FILE_PATH = '/Users/abhudaysingh/Downloads/ICASSP metrics - Sheet7.csv'

# Load and prepare data
df = pd.read_csv(CSV_FILE_PATH, header=1) # header=1 to use the second row as headers
df.rename(columns={
    '# Params (M)': 'Params',
    'Input Size': 'Input_Channels',
    'Theoretical GFLops': 'Theoretical_FLOPS',  # GFLOPs
    'PTFLOPS': 'Actual_FLOPS',  # GFLOPs
    'F.Pass Activation Size (MB)': 'Activation_Size',  # MB
    'Practical Latency (ms)': 'Actual_Latency'  # ms
}, inplace=True)

# Select necessary numeric columns and 'Input_Channels' for color-coding
numeric_cols_for_plot = ['Theoretical_FLOPS', 'Activation_Size', 'Actual_Latency']
cols_to_keep = numeric_cols_for_plot + ['Input_Channels']
df_cleaned = df[cols_to_keep].copy()

# Convert numeric columns to numeric type, coercing errors to NaN
for col in numeric_cols_for_plot:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Drop rows where ANY of the essential numeric columns for plotting are NaN
df_cleaned.dropna(subset=numeric_cols_for_plot, inplace=True)

# Apply log10 transformation
epsilon = 1e-9  # To avoid log(0) if any values are zero or negative
for col in numeric_cols_for_plot:
    # Ensure values are positive before log transformation
    df_cleaned[col] = df_cleaned[col].apply(lambda x: x if x > 0 else epsilon)
    df_cleaned[col] = np.log10(df_cleaned[col])


# Create a mapping for consistent colors based on Input_Channels
unique_input_channels = sorted(df_cleaned['Input_Channels'].unique()) # Sort for consistent legend order
color_map = {
    input_val: px.colors.qualitative.Dark24[i % len(px.colors.qualitative.Dark24)]
    for i, input_val in enumerate(unique_input_channels)
}
df_cleaned['Point_Color'] = df_cleaned['Input_Channels'].map(color_map)


# Define a function to create 2D plots with trendlines, extrapolation, and legend
def create_2d_log_plot_with_legend_and_extrapolation(
        df_plot_full, x_col, y_col, degree=2, extrapolation_factor=0.25
    ):
    """
    Creates a 2D scatter plot with a polynomial trendline, extrapolated lines,
    and a legend for input resolutions.
    """
    if df_plot_full.empty or len(df_plot_full) < degree + 1:
        print(f"Skipping plot for {y_col} vs {x_col}: Insufficient data points after cleaning.")
        return go.Figure()

    # Prepare data for regression (using all data for the trendline, not per color)
    X_fit = df_plot_full[[x_col]]
    y_fit = df_plot_full[y_col]

    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    model.fit(X_fit, y_fit)

    # Define ranges for trendline and extrapolation
    x_min_data, x_max_data = df_plot_full[x_col].min(), df_plot_full[x_col].max()
    x_range_data = np.linspace(x_min_data, x_max_data, 100)
    y_pred_data = model.predict(x_range_data.reshape(-1, 1))

    data_span = x_max_data - x_min_data
    if data_span == 0: data_span = abs(x_min_data * 0.1) if x_min_data != 0 else 0.1 # Handle single point case
    extrapolate_amount = data_span * extrapolation_factor

    x_range_extrapolate_before = np.linspace(x_min_data - extrapolate_amount, x_min_data, 50)
    y_pred_extrapolate_before = model.predict(x_range_extrapolate_before.reshape(-1, 1))

    x_range_extrapolate_after = np.linspace(x_max_data, x_max_data + extrapolate_amount, 50)
    y_pred_extrapolate_after = model.predict(x_range_extrapolate_after.reshape(-1, 1))

    r2 = r2_score(y_fit, model.predict(X_fit))
    fig = go.Figure()

    # Add scatter traces for each Input Channel (for legend and color-coding)
    for channel_name in unique_input_channels: # Use sorted unique_input_channels for legend order
        df_subset = df_plot_full[df_plot_full['Input_Channels'] == channel_name]
        if not df_subset.empty:
            fig.add_trace(go.Scatter(
                x=df_subset[x_col],
                y=df_subset[y_col],
                mode='markers',
                marker=dict(color=color_map[channel_name], size=9, opacity=0.8),
                name=str(channel_name), # Name for the legend item
                legendgroup=str(channel_name) # Group by channel
            ))

    # Add trendline
    fig.add_trace(go.Scatter(
        x=x_range_data, y=y_pred_data, mode='lines',
        name=f'Overall Trend (Deg {degree})', line=dict(color='black', width=3),
        showlegend=True # Show trendline in legend
    ))
    # Add extrapolation lines
    fig.add_trace(go.Scatter(
        x=x_range_extrapolate_before, y=y_pred_extrapolate_before, mode='lines',
        name='Extrapolation', line=dict(color='gray', width=2, dash='dash'),
        showlegend=True, legendgroup='extrapolation' # Group extrapolations
    ))
    fig.add_trace(go.Scatter(
        x=x_range_extrapolate_after, y=y_pred_extrapolate_after, mode='lines',
        name='Extrapolation', line=dict(color='gray', width=2, dash='dash'),
        showlegend=False, legendgroup='extrapolation' # Hide duplicate extrapolation legend
    ))


    # Calculate dynamic axis ranges to ensure all data is visible
    all_x_values = np.concatenate([
        df_plot_full[x_col].values, x_range_data,
        x_range_extrapolate_before, x_range_extrapolate_after
    ])
    all_y_values = np.concatenate([
        df_plot_full[y_col].values, y_pred_data,
        y_pred_extrapolate_before, y_pred_extrapolate_after
    ])

    # Remove NaNs or Infs that might have crept in
    all_x_values = all_x_values[np.isfinite(all_x_values)]
    all_y_values = all_y_values[np.isfinite(all_y_values)]

    final_x_range, final_y_range = None, None
    if all_x_values.size > 0 and all_y_values.size > 0:
        min_x, max_x = np.min(all_x_values), np.max(all_x_values)
        min_y, max_y = np.min(all_y_values), np.max(all_y_values)
        padding_x = (max_x - min_x) * 0.05 if (max_x - min_x) > 0 else 0.1
        padding_y = (max_y - min_y) * 0.05 if (max_y - min_y) > 0 else 0.1
        final_x_range = [min_x - padding_x, max_x + padding_x]
        final_y_range = [min_y - padding_y, max_y + padding_y]


    units = {
        'Theoretical_FLOPS': 'GFLOPs',
        'Activation_Size': 'MB',
        'Actual_Latency': 'ms'
    }
    xaxis_title_text = f'Log10({x_col} / {units.get(x_col, "")})'
    yaxis_title_text = f'Log10({y_col} / {units.get(y_col, "")})'

    # Specific axis range handling for latency
    if y_col == 'Actual_Latency':
        # Desired log-transformed min/max for latency
        desired_y_min_log = np.log10(0.1 + epsilon)
        desired_y_max_log = np.log10(25 + epsilon)
        # Ensure current dynamic range includes these, or expand to include them
        current_min_y = final_y_range[0] if final_y_range else desired_y_min_log
        current_max_y = final_y_range[1] if final_y_range else desired_y_max_log
        final_y_range = [min(current_min_y, desired_y_min_log), max(current_max_y, desired_y_max_log)]

    if x_col == 'Actual_Latency':
        desired_x_min_log = np.log10(0.1 + epsilon)
        desired_x_max_log = np.log10(25 + epsilon)
        current_min_x = final_x_range[0] if final_x_range else desired_x_min_log
        current_max_x = final_x_range[1] if final_x_range else desired_x_max_log
        final_x_range = [min(current_min_x, desired_x_min_log), max(current_max_x, desired_x_max_log)]


    fig.update_layout(
        title=f'Log10({y_col}) vs Log10({x_col})<br>(R² on actual data = {r2:.3f})',
        xaxis_title=xaxis_title_text,
        yaxis_title=yaxis_title_text,
        xaxis_range=final_x_range,
        yaxis_range=final_y_range,
        legend_title_text='Input Resolution / Line Type',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0, # Position legend at the top of the plot area
            xanchor="left", # Anchor to the left of where x is specified
            x=1.02, # Position legend slightly to the right of the plot
            bgcolor='rgba(255,255,255,0.7)', # Semi-transparent background
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1
        ),
        template="plotly_white",
        margin=dict(r=200) # Add right margin to make space for a wide legend
    )
    return fig

# --- Create and Show the Plots ---

# 1. Theoretical_FLOPS vs Activation_Size
fig1 = create_2d_log_plot_with_legend_and_extrapolation(
    df_cleaned, 'Theoretical_FLOPS', 'Activation_Size'
)
fig1.show()

# 2. Theoretical_FLOPS vs Actual_Latency
fig2 = create_2d_log_plot_with_legend_and_extrapolation(
    df_cleaned, 'Theoretical_FLOPS', 'Actual_Latency'
)
fig2.show()

# 3. Activation_Size vs Actual_Latency
fig3 = create_2d_log_plot_with_legend_and_extrapolation(
    df_cleaned, 'Activation_Size', 'Actual_Latency'
)
fig3.show()

print("--- Script finished ---")
