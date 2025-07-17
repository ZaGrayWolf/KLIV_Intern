import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Use the provided file path for your CSV.
CSV_FILE_PATH = '/Users/abhudaysingh/Downloads/ICASSP metrics - Sheet7.csv'

# Load and prepare data
df = pd.read_csv(CSV_FILE_PATH, header=1)
df.rename(columns={
    '# Params (M)': 'Params',
    'Input Size': 'Input_Channels',
    'Theoretical GFLops': 'Theoretical_FLOPS',  # GFLOPs
    'PTFLOPS': 'Actual_FLOPS',  # GFLOPs
    'F.Pass Activation Size (MB)': 'Activation_Size',  # MB
    'Practical Latency (ms)': 'Actual_Latency'  # ms
}, inplace=True)

# Select necessary numeric columns and 'Input_Channels'
numeric_cols_for_plot = ['Theoretical_FLOPS', 'Activation_Size', 'Actual_Latency']
cols_to_keep = numeric_cols_for_plot + ['Input_Channels']
df_cleaned = df[cols_to_keep].copy()

# Convert to numeric, coercing errors to NaN
for col in numeric_cols_for_plot:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Drop rows where ANY of these essential numeric columns are NaN
df_cleaned.dropna(subset=numeric_cols_for_plot, inplace=True)

# Apply log10 transformation
epsilon = 1e-9
for col in numeric_cols_for_plot:
    df_cleaned[col] = df_cleaned[col].apply(lambda x: x if x > 0 else epsilon)
    df_cleaned[col] = np.log10(df_cleaned[col]) # Base-10 log

# Define target points and styles based on the metric
target_latency_values = [25, 10, 1, 0.1]
target_latency_logs = [np.log10(val + epsilon) for val in target_latency_values]
latency_styles = [
    {'symbol': 'square', 'color': '#636EFA', 'size': 10, 'line_dash': 'solid', 'line_opacity': 0.4},
    {'symbol': 'circle', 'color': '#EF553B', 'size': 10, 'line_dash': 'dash', 'line_opacity': 0.4},
    {'symbol': 'diamond', 'color': '#00CC96', 'size': 11, 'line_dash': 'dot', 'line_opacity': 0.4},
    {'symbol': 'cross', 'color': '#AB63FA', 'size': 12, 'line_dash': 'dashdot', 'line_opacity': 0.4}
]

target_activation_values = [1000, 500, 100, 10]
target_activation_logs = [np.log10(val + epsilon) for val in target_activation_values]
activation_styles = [
    {'symbol': 'square', 'color': '#FFA15A', 'size': 10, 'line_dash': 'solid', 'line_opacity': 0.4},
    {'symbol': 'circle', 'color': '#19D3F3', 'size': 10, 'line_dash': 'dash', 'line_opacity': 0.4},
    {'symbol': 'diamond', 'color': '#FF6692', 'size': 11, 'line_dash': 'dot', 'line_opacity': 0.4},
    {'symbol': 'star', 'color': '#B6E880', 'size': 12, 'line_dash': 'dashdot', 'line_opacity': 0.4}
]

# Create a mapping for consistent colors for Input_Channels trendlines and data points
unique_input_channels = sorted(df_cleaned['Input_Channels'].unique())
channel_color_map = {
    input_val: px.colors.qualitative.T10[i % len(px.colors.qualitative.T10)]
    for i, input_val in enumerate(unique_input_channels)
}

# Main plotting function
def create_2d_log_plot_with_projections(
    df_plot_full, x_col, y_col, degree=2
):
    if df_plot_full.empty or len(df_plot_full['Input_Channels'].unique()) < 1:
        print(f"Skipping plot for {y_col} vs {x_col}: Insufficient data.")
        return go.Figure()

    fig = go.Figure()
    all_x_coords_on_plot = []
    all_y_coords_on_plot = []

    # Determine which target points to use for this plot
    is_latency_plot = (x_col == 'Actual_Latency' or y_col == 'Actual_Latency')
    is_activation_plot = (x_col == 'Activation_Size' or y_col == 'Activation_Size')

    # --- PASS 1: Add dummy traces for legend ---
    if is_latency_plot:
        for i, lat_val in enumerate(target_latency_values):
            style = latency_styles[i]
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                name=f'{lat_val}ms Target', legendgroup=f'TargetLatency-{lat_val}ms',
                marker=dict(symbol=style['symbol'], color=style['color'], size=style['size'])))
    elif is_activation_plot:
        for i, act_val in enumerate(target_activation_values):
            style = activation_styles[i]
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                name=f'{act_val}MB Target', legendgroup=f'TargetActivation-{act_val}MB',
                marker=dict(symbol=style['symbol'], color=style['color'], size=style['size'])))

    # --- PASS 2: Plot data/trends and collect projection info ---
    projection_points_to_draw = []

    for channel_name in unique_input_channels:
        df_subset = df_plot_full[df_plot_full['Input_Channels'] == channel_name]
        current_channel_color = channel_color_map[channel_name]
        short_channel_name = str(channel_name).split(" x ")[0] if " x " in str(channel_name) else str(channel_name).split(" ")[0]

        if df_subset.empty or len(df_subset) < degree + 1:
            if not df_subset.empty:
                fig.add_trace(go.Scatter(x=df_subset[x_col], y=df_subset[y_col], mode='markers',
                    name=str(channel_name), legendgroup=str(channel_name),
                    marker=dict(color=current_channel_color, size=8, opacity=0.7),
                    hovertext=[f"Input: {channel_name}<br>{x_col}: {10**x_val:.2f}<br>{y_col}: {10**y_val:.2f}" for x_val, y_val in zip(df_subset[x_col], df_subset[y_col])],
                    hoverinfo='text'))
                all_x_coords_on_plot.extend(df_subset[x_col].tolist()); all_y_coords_on_plot.extend(df_subset[y_col].tolist())
            continue

        X_fit = df_subset[[x_col]]; y_fit = df_subset[y_col]
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
        model.fit(X_fit, y_fit)

        # Add raw data points
        fig.add_trace(go.Scatter(x=df_subset[x_col], y=df_subset[y_col], mode='markers',
            name=str(channel_name), legendgroup=str(channel_name),
            marker=dict(color=current_channel_color, size=8, opacity=0.7),
            hovertext=[f"Input: {channel_name}<br>{x_col}: {10**x_val:.2f}<br>{y_col}: {10**y_val:.2f}" for x_val, y_val in zip(df_subset[x_col], df_subset[y_col])],
            hoverinfo='text'))
        all_x_coords_on_plot.extend(df_subset[x_col].tolist()); all_y_coords_on_plot.extend(df_subset[y_col].tolist())

        # Add main trendline (solid, within data range)
        x_min_subset, x_max_subset = df_subset[x_col].min(), df_subset[x_col].max()
        x_range_subset_trend = np.linspace(x_min_subset, x_max_subset, 100)
        y_pred_subset_trend = model.predict(x_range_subset_trend.reshape(-1, 1))
        fig.add_trace(go.Scatter(x=x_range_subset_trend, y=y_pred_subset_trend, mode='lines',
            name=f'Trend ({short_channel_name})', legendgroup=str(channel_name),
            line=dict(color=current_channel_color, width=2.5)))
        all_x_coords_on_plot.extend(x_range_subset_trend.tolist()); all_y_coords_on_plot.extend(y_pred_subset_trend.tolist())

        # **UPDATED: Extended extrapolation to ensure all target points are reachable**
        targets_log, targets_orig, styles, type_str = [], [], [], ""
        if is_latency_plot:
            targets_log, targets_orig, styles, type_str = target_latency_logs, target_latency_values, latency_styles, "Latency"
        elif is_activation_plot:
            targets_log, targets_orig, styles, type_str = target_activation_logs, target_activation_values, activation_styles, "Activation"
        
        if targets_log:
            overall_x_min, overall_x_max = df_plot_full[x_col].min(), df_plot_full[x_col].max()
            
            # Calculate the minimum extension needed to reach our target points
            min_target_log = min(targets_log)
            max_target_log = max(targets_log)
            
            # **FIXED: More generous extension to ensure all targets are reachable**
            if y_col.endswith(type_str) or (y_col == 'Activation_Size' and type_str == "Activation"):
                # Targets are on Y-axis, extend X more generously
                data_span = overall_x_max - overall_x_min
                extension_factor = 0.8  # Increased from 0.3 to 0.8 for better coverage
                x_range_extended = np.linspace(overall_x_min - data_span * extension_factor, 
                                             overall_x_max + data_span * extension_factor, 500)
            elif x_col.endswith(type_str) or (x_col == 'Activation_Size' and type_str == "Activation"):
                # Targets are on X-axis, extend to cover target range with buffer
                buffer = 0.3  # Increased buffer
                x_range_extended = np.linspace(min(overall_x_min, min_target_log - buffer), 
                                             max(overall_x_max, max_target_log + buffer), 500)
            else:
                continue  # No targets for this axis combination
            
            y_pred_extended = model.predict(x_range_extended.reshape(-1, 1))
            
            # Only show extrapolated portions that extend beyond actual data
            # Before data range
            x_before_mask = x_range_extended < x_min_subset
            if np.any(x_before_mask):
                x_before = x_range_extended[x_before_mask]
                y_before = y_pred_extended[x_before_mask]
                fig.add_trace(go.Scatter(x=x_before, y=y_before, mode='lines',
                    name=f'Extrapolated ({short_channel_name})', legendgroup=str(channel_name),
                    line=dict(color=current_channel_color, width=1.5, dash='dash'),
                    showlegend=False))
                all_x_coords_on_plot.extend(x_before.tolist()); all_y_coords_on_plot.extend(y_before.tolist())
            
            # After data range
            x_after_mask = x_range_extended > x_max_subset
            if np.any(x_after_mask):
                x_after = x_range_extended[x_after_mask]
                y_after = y_pred_extended[x_after_mask]
                fig.add_trace(go.Scatter(x=x_after, y=y_after, mode='lines',
                    name=f'Extrapolated ({short_channel_name})', legendgroup=str(channel_name),
                    line=dict(color=current_channel_color, width=1.5, dash='dash'),
                    showlegend=False))
                all_x_coords_on_plot.extend(x_after.tolist()); all_y_coords_on_plot.extend(y_after.tolist())

            # Collect projection points
            if y_col.endswith(type_str) or (y_col == 'Activation_Size' and type_str == "Activation"):
                for i, target_y_log in enumerate(targets_log):
                    idx = (np.abs(y_pred_extended - target_y_log)).argmin()
                    projection_points_to_draw.append({
                        'x': x_range_extended[idx], 'y': target_y_log, 'style_idx': i, 
                        'styles_list': styles, 'target_val': targets_orig[i], 'type_str': type_str,
                        'channel_name': channel_name, 'short_channel_name': short_channel_name,
                        'channel_color': current_channel_color
                    })
                    all_x_coords_on_plot.append(x_range_extended[idx]); all_y_coords_on_plot.append(target_y_log)
            elif x_col.endswith(type_str) or (x_col == 'Activation_Size' and type_str == "Activation"):
                for i, target_x_log in enumerate(targets_log):
                    idx = (np.abs(x_range_extended - target_x_log)).argmin()
                    projection_points_to_draw.append({
                        'x': target_x_log, 'y': y_pred_extended[idx], 'style_idx': i, 
                        'styles_list': styles, 'target_val': targets_orig[i], 'type_str': type_str,
                        'channel_name': channel_name, 'short_channel_name': short_channel_name,
                        'channel_color': current_channel_color
                    })
                    all_x_coords_on_plot.append(target_x_log); all_y_coords_on_plot.append(y_pred_extended[idx])

    # --- PASS 3: Calculate final axis ranges with guaranteed target visibility ---
    units = {'Theoretical_FLOPS': 'GFLOPs', 'Activation_Size': 'MB', 'Actual_Latency': 'ms'}
    xaxis_title_text = f'Log10({x_col} / {units.get(x_col, "")})'
    yaxis_title_text = f'Log10({y_col} / {units.get(y_col, "")})'
    final_x_range_calc, final_y_range_calc = None, None

    if all_x_coords_on_plot and all_y_coords_on_plot:
        valid_x = np.array([x for x in all_x_coords_on_plot if np.isfinite(x)])
        valid_y = np.array([y for y in all_y_coords_on_plot if np.isfinite(y)])
        if valid_x.size > 0 and valid_y.size > 0:
            min_x_plot, max_x_plot = np.min(valid_x), np.max(valid_x)
            min_y_plot, max_y_plot = np.min(valid_y), np.max(valid_y)
            # **INCREASED padding to ensure visibility**
            padding_x = (max_x_plot - min_x_plot) * 0.15 if (max_x_plot - min_x_plot) > 0 else 0.5
            padding_y = (max_y_plot - min_y_plot) * 0.15 if (max_y_plot - min_y_plot) > 0 else 0.5
            final_x_range_calc = [min_x_plot - padding_x, max_x_plot + padding_x]
            final_y_range_calc = [min_y_plot - padding_y, max_y_plot + padding_y]

    # **CRITICAL FIX: Force inclusion of all target points in axis ranges**
    if is_latency_plot:
        # Ensure ALL latency targets are visible
        target_min_log = min(target_latency_logs) - 0.1
        target_max_log = max(target_latency_logs) + 0.1
        if y_col == 'Actual_Latency':
            if final_y_range_calc:
                final_y_range_calc[0] = min(final_y_range_calc[0], target_min_log)
                final_y_range_calc[1] = max(final_y_range_calc[1], target_max_log)
            else:
                final_y_range_calc = [target_min_log, target_max_log]
        if x_col == 'Actual_Latency':
            if final_x_range_calc:
                final_x_range_calc[0] = min(final_x_range_calc[0], target_min_log)
                final_x_range_calc[1] = max(final_x_range_calc[1], target_max_log)
            else:
                final_x_range_calc = [target_min_log, target_max_log]
                
    if is_activation_plot:
        # Ensure ALL activation targets are visible
        target_min_log = min(target_activation_logs) - 0.1
        target_max_log = max(target_activation_logs) + 0.1
        if y_col == 'Activation_Size':
            if final_y_range_calc:
                final_y_range_calc[0] = min(final_y_range_calc[0], target_min_log)
                final_y_range_calc[1] = max(final_y_range_calc[1], target_max_log)
            else:
                final_y_range_calc = [target_min_log, target_max_log]
        if x_col == 'Activation_Size':
            if final_x_range_calc:
                final_x_range_calc[0] = min(final_x_range_calc[0], target_min_log)
                final_x_range_calc[1] = max(final_x_range_calc[1], target_max_log)
            else:
                final_x_range_calc = [target_min_log, target_max_log]

    # Fallback
    if final_x_range_calc is None: final_x_range_calc = [df_plot_full[x_col].min()-0.5, df_plot_full[x_col].max()+0.5]
    if final_y_range_calc is None: final_y_range_calc = [df_plot_full[y_col].min()-0.5, df_plot_full[y_col].max()+0.5]

    # --- PASS 4: Draw projection markers and lines using final axis ranges ---
    if projection_points_to_draw:
        for marker_info in projection_points_to_draw:
            proj_x, proj_y = marker_info['x'], marker_info['y']
            style_idx, styles_list = marker_info['style_idx'], marker_info['styles_list']
            target_val, type_str = marker_info['target_val'], marker_info['type_str']
            channel_name, short_ch_name = marker_info['channel_name'], marker_info['short_channel_name']
            channel_color = marker_info['channel_color']
            
            style = styles_list[style_idx]
            hex_color, opacity, dash_style = style['color'], style.get('line_opacity', 0.4), style['line_dash']
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            line_color_rgba = f"rgba({r}, {g}, {b}, {opacity})"
            unit = "ms" if type_str == "Latency" else "MB"

            # Add marker on curve
            fig.add_trace(go.Scatter(x=[proj_x], y=[proj_y], mode='markers',
                name=f'{target_val}{unit} ({short_ch_name})', 
                legendgroup=f'Target{type_str}-{target_val}{unit}',
                marker=dict(symbol=style['symbol'], color=style['color'], size=style['size'],
                           line=dict(width=1, color=channel_color)),
                hovertext=f"Input: {channel_name}<br>Target {type_str}: {target_val}{unit}<br>Extrapolated Point: ({10**proj_x:.2f}, {10**proj_y:.2f})",
                hoverinfo='text', showlegend=False))

            # Horizontal line to y-axis
            fig.add_shape(type="line", layer="below",
                          x0=final_x_range_calc[0], y0=proj_y, x1=proj_x, y1=proj_y,
                          line=dict(color=line_color_rgba, width=1.5, dash=dash_style))
            # Vertical line to x-axis
            fig.add_shape(type="line", layer="below",
                          x0=proj_x, y0=final_y_range_calc[0], x1=proj_x, y1=proj_y,
                          line=dict(color=line_color_rgba, width=1.5, dash=dash_style))
    
    fig.update_layout(
        title=f'Log10({y_col}) vs Log10({x_col})',
        xaxis_title=xaxis_title_text, yaxis_title=yaxis_title_text,
        xaxis_range=final_x_range_calc, yaxis_range=final_y_range_calc,
        showlegend=True, template="plotly_white",
        legend=dict(title_text='<b>Input Res. / Target Value</b>', orientation="v",
                    yanchor="top", y=1.0, xanchor="left", x=1.02,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.5)',
                    borderwidth=1, tracegroupgap=10, itemsizing='constant'),
        margin=dict(r=400))
    return fig

# --- Create and Show the Plots ---
print("Generating FLOPS vs Activation_Size plot...")
fig1 = create_2d_log_plot_with_projections(df_cleaned, 'Theoretical_FLOPS', 'Activation_Size')
fig1.show()
print("\nGenerating FLOPS vs Actual_Latency plot...")
fig2 = create_2d_log_plot_with_projections(df_cleaned, 'Theoretical_FLOPS', 'Actual_Latency')
fig2.show()
print("\nGenerating Activation_Size vs Actual_Latency plot...")
fig3 = create_2d_log_plot_with_projections(df_cleaned, 'Activation_Size', 'Actual_Latency')
fig3.show()
print("--- Script finished ---")
 
#Log Graphs of all 3