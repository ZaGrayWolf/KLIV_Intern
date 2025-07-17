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
df_original = df[cols_to_keep].copy()

# Convert to numeric, coercing errors to NaN
for col in numeric_cols_for_plot:
    df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

# Drop rows where ANY of these essential numeric columns are NaN
df_original = df_original.dropna(subset=numeric_cols_for_plot)

# Define target points and styles based on the metric
target_latency_values = [25, 10, 1, 0.1]
latency_styles = [
    {'symbol': 'square', 'color': '#636EFA', 'size': 10, 'line_dash': 'solid', 'line_opacity': 0.4},
    {'symbol': 'circle', 'color': '#EF553B', 'size': 10, 'line_dash': 'dash', 'line_opacity': 0.4},
    {'symbol': 'diamond', 'color': '#00CC96', 'size': 11, 'line_dash': 'dot', 'line_opacity': 0.4},
    {'symbol': 'cross', 'color': '#AB63FA', 'size': 12, 'line_dash': 'dashdot', 'line_opacity': 0.4}
]

target_activation_values = [1000, 500, 100, 10]
activation_styles = [
    {'symbol': 'square', 'color': '#FFA15A', 'size': 10, 'line_dash': 'solid', 'line_opacity': 0.4},
    {'symbol': 'circle', 'color': '#19D3F3', 'size': 10, 'line_dash': 'dash', 'line_opacity': 0.4},
    {'symbol': 'diamond', 'color': '#FF6692', 'size': 11, 'line_dash': 'dot', 'line_opacity': 0.4},
    {'symbol': 'star', 'color': '#B6E880', 'size': 12, 'line_dash': 'dashdot', 'line_opacity': 0.4}
]

# Get unique input channels
unique_input_channels = sorted(df_original['Input_Channels'].unique())

# Create color mapping for networks
channel_color_map = {
    input_val: px.colors.qualitative.T10[i % len(px.colors.qualitative.T10)]
    for i, input_val in enumerate(unique_input_channels)
}

# Main plotting function with scale option
def create_plot_for_input_size(df_plot_full, input_size, x_col, y_col, log_scale=True, degree=2):
    """Create plot for specific input size with given scale"""
    
    # Filter data for specific input size
    df_filtered = df_plot_full[df_plot_full['Input_Channels'] == input_size].copy()
    
    if df_filtered.empty:
        print(f"No data for input size {input_size}")
        return go.Figure()
    
    # Apply log transformation if needed
    df_plot = df_filtered.copy()
    epsilon = 1e-9
    
    if log_scale:
        for col in [x_col, y_col]:
            df_plot[col] = df_plot[col].apply(lambda x: x if x > 0 else epsilon)
            df_plot[col] = np.log10(df_plot[col])
        scale_suffix = " (Log Scale)"
        target_latency_logs = [np.log10(val + epsilon) for val in target_latency_values]
        target_activation_logs = [np.log10(val + epsilon) for val in target_activation_values]
    else:
        scale_suffix = " (Linear Scale)"
        target_latency_logs = target_latency_values
        target_activation_logs = target_activation_values

    fig = go.Figure()
    all_x_coords = []
    all_y_coords = []

    # Determine which target points to use
    is_latency_plot = (x_col == 'Actual_Latency' or y_col == 'Actual_Latency')
    is_activation_plot = (x_col == 'Activation_Size' or y_col == 'Activation_Size')

    # Add dummy traces for legend
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

    projection_points_to_draw = []

    # Process each network for this input size
    networks_in_input = df_plot['Network'].unique() if 'Network' in df_plot.columns else ['Combined']
    
    for network in networks_in_input:
        if 'Network' in df_plot.columns:
            df_subset = df_plot[df_plot['Network'] == network]
            color = channel_color_map.get(network, '#1f77b4')
            name = network
        else:
            df_subset = df_plot
            color = channel_color_map.get(input_size, '#1f77b4')
            name = str(input_size)

        if len(df_subset) < degree + 1:
            # Plot points even if no trendline
            if not df_subset.empty:
                fig.add_trace(go.Scatter(
                    x=df_subset[x_col], y=df_subset[y_col], mode='markers',
                    name=name, marker=dict(color=color, size=8, opacity=0.7)))
                all_x_coords.extend(df_subset[x_col].tolist())
                all_y_coords.extend(df_subset[y_col].tolist())
            continue

        # Fit model
        X_fit = df_subset[[x_col]]
        y_fit = df_subset[y_col]
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
        model.fit(X_fit, y_fit)

        # Add data points
        fig.add_trace(go.Scatter(
            x=df_subset[x_col], y=df_subset[y_col], mode='markers',
            name=name, marker=dict(color=color, size=8, opacity=0.7)))
        all_x_coords.extend(df_subset[x_col].tolist())
        all_y_coords.extend(df_subset[y_col].tolist())

        # Add main trendline
        x_min, x_max = df_subset[x_col].min(), df_subset[x_col].max()
        x_range_trend = np.linspace(x_min, x_max, 100)
        y_pred_trend = model.predict(x_range_trend.reshape(-1, 1))
        
        fig.add_trace(go.Scatter(
            x=x_range_trend, y=y_pred_trend, mode='lines',
            name=f'Trend ({name})', line=dict(color=color, width=2.5),
            showlegend=False))
        all_x_coords.extend(x_range_trend.tolist())
        all_y_coords.extend(y_pred_trend.tolist())

        # Extended extrapolation
        targets_log, targets_orig, styles, type_str = [], [], [], ""
        if is_latency_plot:
            targets_log, targets_orig, styles, type_str = target_latency_logs, target_latency_values, latency_styles, "Latency"
        elif is_activation_plot:
            targets_log, targets_orig, styles, type_str = target_activation_logs, target_activation_values, activation_styles, "Activation"
        
        if targets_log:
            # Create extended range
            data_span = x_max - x_min
            extension_factor = 0.8
            x_range_extended = np.linspace(x_min - data_span * extension_factor, 
                                         x_max + data_span * extension_factor, 500)
            y_pred_extended = model.predict(x_range_extended.reshape(-1, 1))
            
            # Add extrapolated lines in BLACK
            x_before_mask = x_range_extended < x_min
            if np.any(x_before_mask):
                fig.add_trace(go.Scatter(
                    x=x_range_extended[x_before_mask], y=y_pred_extended[x_before_mask], 
                    mode='lines', name=f'Extrapolated ({name})',
                    line=dict(color='black', width=1.5, dash='dash'), showlegend=False))
                all_x_coords.extend(x_range_extended[x_before_mask].tolist())
                all_y_coords.extend(y_pred_extended[x_before_mask].tolist())
            
            x_after_mask = x_range_extended > x_max
            if np.any(x_after_mask):
                fig.add_trace(go.Scatter(
                    x=x_range_extended[x_after_mask], y=y_pred_extended[x_after_mask], 
                    mode='lines', name=f'Extrapolated ({name})',
                    line=dict(color='black', width=1.5, dash='dash'), showlegend=False))
                all_x_coords.extend(x_range_extended[x_after_mask].tolist())
                all_y_coords.extend(y_pred_extended[x_after_mask].tolist())

            # Collect projection points
            if y_col.endswith(type_str) or (y_col == 'Activation_Size' and type_str == "Activation"):
                for i, target_y_log in enumerate(targets_log):
                    idx = (np.abs(y_pred_extended - target_y_log)).argmin()
                    projection_points_to_draw.append({
                        'x': x_range_extended[idx], 'y': target_y_log, 'style_idx': i,
                        'styles_list': styles, 'target_val': targets_orig[i], 'type_str': type_str,
                        'network': name
                    })
                    all_x_coords.append(x_range_extended[idx])
                    all_y_coords.append(target_y_log)
                    
            elif x_col.endswith(type_str) or (x_col == 'Activation_Size' and type_str == "Activation"):
                for i, target_x_log in enumerate(targets_log):
                    idx = (np.abs(x_range_extended - target_x_log)).argmin()
                    projection_points_to_draw.append({
                        'x': target_x_log, 'y': y_pred_extended[idx], 'style_idx': i,
                        'styles_list': styles, 'target_val': targets_orig[i], 'type_str': type_str,
                        'network': name
                    })
                    all_x_coords.append(target_x_log)
                    all_y_coords.append(y_pred_extended[idx])

    # Calculate axis ranges - LIMIT TO MAX VALUES
    if all_x_coords and all_y_coords:
        valid_x = np.array([x for x in all_x_coords if np.isfinite(x)])
        valid_y = np.array([y for y in all_y_coords if np.isfinite(y)])
        
        if valid_x.size > 0 and valid_y.size > 0:
            min_x, max_x = np.min(valid_x), np.max(valid_x)
            min_y, max_y = np.min(valid_y), np.max(valid_y)
            
            # Add small padding but limit to max values
            padding_x = (max_x - min_x) * 0.05
            padding_y = (max_y - min_y) * 0.05
            final_x_range = [min_x - padding_x, max_x + padding_x]
            final_y_range = [min_y - padding_y, max_y + padding_y]
        else:
            final_x_range = [0, 1]
            final_y_range = [0, 1]
    else:
        final_x_range = [0, 1]
        final_y_range = [0, 1]

    # Draw projection markers and intercept lines
    if projection_points_to_draw:
        for marker_info in projection_points_to_draw:
            proj_x, proj_y = marker_info['x'], marker_info['y']
            style_idx = marker_info['style_idx']
            styles_list = marker_info['styles_list']
            target_val = marker_info['target_val']
            type_str = marker_info['type_str']
            
            style = styles_list[style_idx]
            unit = "ms" if type_str == "Latency" else "MB"

            # Add marker on curve
            fig.add_trace(go.Scatter(
                x=[proj_x], y=[proj_y], mode='markers',
                name=f'{target_val}{unit}',
                marker=dict(symbol=style['symbol'], color=style['color'], size=style['size']),
                showlegend=False))

            # Add BLACK intercept lines with value annotations
            # Horizontal line to y-axis
            fig.add_shape(type="line", layer="below",
                          x0=final_x_range[0], y0=proj_y, x1=proj_x, y1=proj_y,
                          line=dict(color='black', width=1.5, dash='dot'))
            
            # Vertical line to x-axis
            fig.add_shape(type="line", layer="below",
                          x0=proj_x, y0=final_y_range[0], x1=proj_x, y1=proj_y,
                          line=dict(color='black', width=1.5, dash='dot'))
            
            # Add BLACK annotations for intercept values
            # Y-axis intercept annotation
            if log_scale:
                y_intercept_value = 10**proj_y if type_str == "Latency" else 10**proj_y
                x_intercept_value = 10**proj_x if type_str != "Latency" else 10**proj_x
            else:
                y_intercept_value = proj_y
                x_intercept_value = proj_x
            
            fig.add_annotation(
                x=final_x_range[0], y=proj_y,
                text=f"{target_val}{unit}",
                showarrow=False,
                xanchor="right",
                font=dict(color="black", size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            # X-axis intercept annotation
            x_unit = "GFLOPs" if "FLOPS" in x_col else ("MB" if "Activation" in x_col else "ms")
            fig.add_annotation(
                x=proj_x, y=final_y_range[0],
                text=f"{x_intercept_value:.1f}{x_unit}",
                showarrow=False,
                yanchor="top",
                font=dict(color="black", size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

    # Update layout
    units = {'Theoretical_FLOPS': 'GFLOPs', 'Activation_Size': 'MB', 'Actual_Latency': 'ms'}
    
    if log_scale:
        xaxis_title = f'Log10({x_col} / {units.get(x_col, "")})'
        yaxis_title = f'Log10({y_col} / {units.get(y_col, "")})'
    else:
        xaxis_title = f'{x_col} ({units.get(x_col, "")})'
        yaxis_title = f'{y_col} ({units.get(y_col, "")})'

    fig.update_layout(
        title=f'<b>{y_col} vs {x_col}</b><br><sub>Input Size: {input_size}{scale_suffix}</sub>',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_range=final_x_range,
        yaxis_range=final_y_range,
        showlegend=True,
        template="plotly_white",
        legend=dict(
            title_text='<b>Networks / Target Values</b>',
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1,
            tracegroupgap=10,
            itemsizing='constant'
        ),
        margin=dict(r=300),
        height=600
    )
    
    return fig

# Generate all plots
plot_combinations = [
    ('Theoretical_FLOPS', 'Activation_Size'),
    ('Theoretical_FLOPS', 'Actual_Latency'), 
    ('Activation_Size', 'Actual_Latency')
]

print("Generating plots for each input size with both log and linear scales...")
print("=" * 60)

for input_size in unique_input_channels:
    print(f"\nGenerating plots for Input Size: {input_size}")
    
    for x_col, y_col in plot_combinations:
        # Log scale version
        print(f"  Creating {y_col} vs {x_col} (Log Scale)...")
        fig_log = create_plot_for_input_size(df_original, input_size, x_col, y_col, log_scale=True)
        fig_log.show()
        
        # Linear scale version
        print(f"  Creating {y_col} vs {x_col} (Linear Scale)...")
        fig_linear = create_plot_for_input_size(df_original, input_size, x_col, y_col, log_scale=False)
        fig_linear.show()

print("\n" + "=" * 60)
print("All plots generated successfully!")
print(f"Total plots created: {len(unique_input_channels) * len(plot_combinations) * 2}")
print(f"Input sizes processed: {unique_input_channels}")
#Has 48 graphs run causiously