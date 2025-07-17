import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Use the provided file path for your CSV
CSV_FILE_PATH = '/Users/abhudaysingh/Downloads/ICASSP metrics - Sheet7.csv'

# Load and prepare data
df = pd.read_csv(CSV_FILE_PATH, header=1)
df.rename(columns={
    '# Params (M)': 'Params',
    'Input Size': 'Input_Channels',
    'Theoretical GFLops': 'Theoretical_FLOPS',
    'PTFLOPS': 'Actual_FLOPS',
    'F.Pass Activation Size (MB)': 'Activation_Size',
    'Practical Latency (ms)': 'Actual_Latency'
}, inplace=True)

# Select relevant columns and clean data
cols = ['Network', 'Input_Channels', 'Params', 'Activation_Size', 'Theoretical_FLOPS', 'Actual_FLOPS', 'Actual_Latency']
df_table = df[cols].copy()

# Convert to numeric where needed
numeric_cols = ['Params', 'Activation_Size', 'Theoretical_FLOPS', 'Actual_FLOPS', 'Actual_Latency']
for col in numeric_cols:
    df_table[col] = pd.to_numeric(df_table[col], errors='coerce')

# Drop rows with missing values
expected_values_table = df_table.dropna().copy()

# Display the table
print("Expected Values Table for Input Size - Network Pairs:")
print("=" * 80)
print(expected_values_table.to_string(index=False))
print("\n")

# Create the activation size vs input data size plot
fig = go.Figure()

# Get unique networks for color mapping
unique_networks = expected_values_table['Network'].unique()
colors = px.colors.qualitative.Set1[:len(unique_networks)]

# Calculate sizeref for proper marker scaling
max_params = expected_values_table['Params'].max()
desired_max_marker_size = 40
sizeref = 2. * max_params / (desired_max_marker_size ** 2)

# Add scatter points for each network
for i, network in enumerate(unique_networks):
    network_data = expected_values_table[expected_values_table['Network'] == network]
    
    fig.add_trace(go.Scatter(
        x=network_data['Input_Channels'],
        y=network_data['Activation_Size'],
        mode='markers',
        name=network,
        marker=dict(
            color=colors[i],
            size=network_data['Params'],
            sizemode='area',
            sizeref=sizeref,  # FIXED: Removed sizemax, added sizeref
            sizemin=8,
            opacity=0.7,
            line=dict(width=1, color='black')
        ),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'Input Size: %{x}<br>' +
                     'Activation Size: %{y:.1f} MB<br>' +
                     'Parameters: %{customdata:.2f}M<br>' +
                     '<extra></extra>',
        customdata=network_data['Params']
    ))

# Device memory limits (convert GB to MB)
device_memory_limits = [
    {'name': 'Laptop RAM (16GB)', 'memory_mb': 16 * 1024, 'line_style': 'dash', 'color': 'black'},
    {'name': 'Raspberry Pi 5 (8GB)', 'memory_mb': 8 * 1024, 'line_style': 'dot', 'color': 'grey'},
    {'name': 'Jetson Nano (4GB)', 'memory_mb': 4 * 1024, 'line_style': 'dashdot', 'color': 'darkgray'},
    {'name': 'Server (96GB)', 'memory_mb': 96 * 1024, 'line_style': 'solid', 'color': 'dimgray'}
]

# Add dummy traces for device memory limits in legend
for device in device_memory_limits:
    fig.add_trace(go.Scatter(
        x=[None], y=[None],  # No actual data points
        mode='lines',
        name=device['name'],
        legendgroup='Device Memory Limits',
        line=dict(
            color=device['color'],
            width=2,
            dash=device['line_style']
        ),
        showlegend=True
    ))

# Add horizontal lines for device memory limits
for device in device_memory_limits:
    fig.add_hline(
        y=device['memory_mb'],
        line_dash=device['line_style'],
        line_color=device['color'],
        line_width=2,
        annotation_text=device['name'],
        annotation_position="top right",
        annotation=dict(
            bgcolor="white",
            bordercolor=device['color'],
            borderwidth=1,
            font=dict(size=10)
        )
    )

# Update layout
fig.update_layout(
    title='<b>Activation Size vs Input Data Size by Network</b><br><sub>Marker size indicates number of parameters | Horizontal lines show device memory limits</sub>',
    xaxis_title='Input Data Size (Resolution)',
    yaxis_title='Activation Size (MB)',
    yaxis_type='log',
    showlegend=True,
    template='plotly_white',
    legend=dict(
        title='<b>Networks & Device Memory Limits</b>',
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        tracegroupgap=10  # Add space between network traces and device limit traces
    ),
    margin=dict(r=250),
    hovermode='closest',
    height=600
)

# Update x-axis to show all input sizes clearly
fig.update_xaxes(tickangle=45)

fig.show()

# Summary statistics and compatibility analysis (same as before)
print("\nSummary Statistics by Network:")
print("=" * 50)
summary = expected_values_table.groupby('Network').agg({
    'Params': ['mean', 'min', 'max'],
    'Activation_Size': ['mean', 'min', 'max'],
    'Theoretical_FLOPS': ['mean', 'min', 'max'],
    'Actual_Latency': ['mean', 'min', 'max']
}).round(2)
print(summary)

# Memory compatibility analysis
print("\nMemory Compatibility Analysis:")
print("=" * 40)
for device in device_memory_limits:
    limit_mb = device['memory_mb']
    device_name = device['name']
    compatible = expected_values_table[expected_values_table['Activation_Size'] <= limit_mb]
    incompatible = expected_values_table[expected_values_table['Activation_Size'] > limit_mb]
    
    print(f"\n{device_name} ({limit_mb/1024:.0f}GB limit):")
    print(f"  Compatible configurations: {len(compatible)}/{len(expected_values_table)}")
    if len(incompatible) > 0:
        print(f"  Incompatible configurations:")
        for _, row in incompatible.iterrows():
            print(f"    - {row['Network']} @ {row['Input_Channels']}: {row['Activation_Size']:.1f}MB")


#The Activation Size vs Input Data Size plot been created successfully