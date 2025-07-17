import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load and prepare data
CSV_FILE_PATH = '/Users/abhudaysingh/Downloads/ICASSP metrics - Sheet7.csv'
df = pd.read_csv(CSV_FILE_PATH, header=1)
df.rename(columns={
    '# Params (M)': 'Params',
    'Input Size': 'Input_Channels',
    'Theoretical GFLops': 'Theoretical_FLOPS',
    'PTFLOPS': 'Actual_FLOPS',
    'F.Pass Activation Size (MB)': 'Activation_Size',
    'Practical Latency (ms)': 'Actual_Latency'
}, inplace=True)

# Filter numeric metrics
cols = ['Theoretical_FLOPS','Activation_Size','Actual_Latency','Input_Channels','Network']
df = df[cols].dropna(subset=['Theoretical_FLOPS','Activation_Size','Actual_Latency'])

# Network color mapping
networks = sorted(df['Network'].unique())
network_color_map = {
    net: px.colors.qualitative.T10[i % len(px.colors.qualitative.T10)]
    for i, net in enumerate(networks)
}

# Benchmark symbols for latency and activation targets
benchmark_symbols = {
    25:   {'symbol':'square','color':'#636EFA','size':12},
    10:   {'symbol':'circle','color':'#EF553B','size':12},
    1:    {'symbol':'cross','color':'#AB63FA','size':12},
    0.1:  {'symbol':'diamond','color':'#00CC96','size':12},
    1000: {'symbol':'star','color':'#FFA15A','size':12},
    500:  {'symbol':'triangle-up','color':'#19D3F3','size':12},
    100:  {'symbol':'hexagon','color':'#FF6692','size':12},
    ('act',10): {'symbol':'circle-open','color':'#2CA02C','size':12}
}

def find_closest_point_on_line(x_point, y_point, x_line, y_line):
    """Find the closest point on a line to a given point."""
    distances = np.sqrt((x_line - x_point)**2 + (y_line - y_point)**2)
    min_idx = np.argmin(distances)
    return x_line[min_idx], y_line[min_idx], min_idx

def plot_for_input(df, size, x_col, y_col, targets, log_scale=False):
    """Generate responsive full-screen Plotly chart with extended trendline and latching points."""
    dff = df[df['Input_Channels']==size]
    if dff.empty:
        return go.Figure()

    eps = 1e-9
    if log_scale:
        X = np.log10(dff[x_col] + eps).values.reshape(-1,1)
        y = np.log10(dff[y_col] + eps).values
        targets_t = [np.log10(t + eps) for t in targets]
        suffix = " (Log)"
    else:
        X = dff[[x_col]].values
        y = dff[y_col].values
        targets_t = targets
        suffix = ""

    fig = go.Figure()

    # Plot network points
    network_points = []  # Store network points for latching
    for net in networks:
        sub = dff[dff['Network']==net]
        if sub.empty: 
            continue
        sx = np.log10(sub[x_col] + eps) if log_scale else sub[x_col]
        sy = np.log10(sub[y_col] + eps) if log_scale else sub[y_col]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode='markers', name=net,
            marker=dict(color=network_color_map[net], size=10, opacity=0.8)
        ))
        
        # Store network points for latching
        for i in range(len(sx)):
            network_points.append((sx.iloc[i], sy.iloc[i], net))

    # Fit polynomial model
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)
    
    # Create EXTENDED trendline (beyond data range)
    x_min, x_max = X.min(), X.max()
    x_range = x_max - x_min
    extension_factor = 1.5  # Extend line by 50% on each side
    
    # Extended range for trendline
    x_extended_min = x_min - x_range * extension_factor
    x_extended_max = x_max + x_range * extension_factor
    
    # Generate extended trendline points
    x_extended = np.linspace(x_extended_min, x_extended_max, 500)
    y_extended = model.predict(x_extended.reshape(-1,1))
    
    # Plot original trendline (within data range)
    x_original = np.linspace(x_min, x_max, 200)
    y_original = model.predict(x_original.reshape(-1,1))
    fig.add_trace(go.Scatter(
        x=x_original, y=y_original, mode='lines',
        line=dict(color='rgba(150,150,255,0.8)', width=4),
        name='Trendline'
    ))
    
    # Plot extended trendline (beyond data range) - BLACK COLOR
    fig.add_trace(go.Scatter(
        x=x_extended, y=y_extended, mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Extended Trendline'
    ))

    # Add targets with drop-down to axes
    target_coords = []
    for t, tt in zip(targets, targets_t):
        idx = np.argmin(np.abs(y_extended - tt))
        px_val, py_val = x_extended[idx], y_extended[idx]
        key = t if not (x_col!='Actual_Latency' and t==10) else ('act',10)
        st = benchmark_symbols.get(key, {'symbol':'circle','color':'black','size':10})
        
        # Store target coordinates
        target_coords.append((px_val, py_val))
        
        # Marker
        fig.add_trace(go.Scatter(
            x=[px_val], y=[py_val], mode='markers',
            marker=dict(symbol=st['symbol'], color=st['color'],
                        size=st['size'], line=dict(width=2, color='black')),
            name=f'{t} Target'
        ))

    # LATCH NETWORK POINTS TO EXTENDED TRENDLINE
    for x_net, y_net, net_name in network_points:
        # Find closest point on extended trendline
        closest_x, closest_y, closest_idx = find_closest_point_on_line(
            x_net, y_net, x_extended, y_extended
        )
        
        # Add latching line from network point to closest point on trendline
        fig.add_shape(
            type='line',
            x0=x_net, y0=y_net,
            x1=closest_x, y1=closest_y,
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.6
        )
        
        # Add small marker at the latching point on trendline
        fig.add_trace(go.Scatter(
            x=[closest_x], y=[closest_y], mode='markers',
            marker=dict(color='red', size=6, symbol='x'),
            name=f'{net_name} Latch',
            showlegend=False
        ))

    # Calculate axis ranges to zoom in only till the end of the extended line
    x_range_zoom = [x_extended_min, x_extended_max]
    y_range_zoom = [min(y_extended), max(y_extended)]
    
    # Add some padding to y-range for better visualization
    y_padding = (y_range_zoom[1] - y_range_zoom[0]) * 0.05
    y_range_zoom = [y_range_zoom[0] - y_padding, y_range_zoom[1] + y_padding]

    # Add drop-down lines using zoomed axis ranges
    for px_val, py_val in target_coords:
        # Vertical drop line to x-axis
        fig.add_shape(type='line', 
                      x0=px_val, y0=y_range_zoom[0],
                      x1=px_val, y1=py_val,
                      line=dict(color='black', width=2, dash='dot'),
                      xref='x', yref='y')
        
        # Horizontal drop line to y-axis
        fig.add_shape(type='line', 
                      x0=x_range_zoom[0], y0=py_val,
                      x1=px_val, y1=py_val,
                      line=dict(color='black', width=2, dash='dot'),
                      xref='x', yref='y')

    # Responsive layout with zoomed ranges
    fig.update_layout(
        title=f'{y_col} vs {x_col}<br><sub>Input Size: {size}{suffix}</sub>',
        xaxis=dict(
            showline=True, 
            zeroline=True, 
            ticks='outside', 
            visible=True,
            range=x_range_zoom,  # Zoomed to extended line range
            title=x_col,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            showline=True, 
            zeroline=True, 
            ticks='outside', 
            visible=True,
            range=y_range_zoom,  # Zoomed to extended line range
            title=y_col,
            linewidth=2,
            linecolor='black'
        ),
        autosize=True, width=None, height=None,
        template='plotly_white',
        legend=dict(orientation='v', y=1, x=1.02,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='black'),
        margin=dict(l=80, r=350, t=120, b=80)
    )
    return fig

# Generate all plots (linear + log) for each input size and metric
input_sizes = sorted(df['Input_Channels'].unique())
configs = [
    ('Theoretical_FLOPS','Actual_Latency',[25,10,1,0.1]),
    ('Theoretical_FLOPS','Activation_Size',[1000,500,100,10]),
    ('Activation_Size','Actual_Latency',[25,10,1,0.1])
]

total_plots = 0
for size in input_sizes:
    for xcol, ycol, targs in configs:
        plot_for_input(df, size, xcol, ycol, targs, log_scale=False).show()
        plot_for_input(df, size, xcol, ycol, targs, log_scale=True).show()
        total_plots += 2

print(f"Total plots generated: {total_plots}")
