import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joypy  # Changed this line
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
# --- Configuration ---
CSV_FILE_PATH = '/Users/abhudaysingh/Downloads/ICASSP metrics - Sheet7.csv' # Make sure this CSV is in the same directory


# --- Load and Prepare Data ---
print(f"--- Loading and Preparing Data from '{CSV_FILE_PATH}' ---")
try:
    df = pd.read_csv(CSV_FILE_PATH, header=1)
except FileNotFoundError:
    print(f"Error: '{CSV_FILE_PATH}' not found.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

column_rename_map = {
    'Unnamed: 0': 'Network',
    '# Params (M)': 'Params',
    'Input Size': 'Input_Channels',
    'Theoretical GFLops': 'Theoretical_FLOPS',
    'PTFLOPS': 'Actual_FLOPS',
    'F.Pass Activation Size (MB)': 'Activation_Size',
    'Latency': 'Theoretical_Latency',
    'Practical Latency (ms)': 'Actual_Latency'
}
df.rename(columns=column_rename_map, inplace=True)
expected_script_columns = list(column_rename_map.values())
try:
    df = df[expected_script_columns]
except KeyError as e:
    print(f"KeyError after renaming: {e}. Available: {df.columns.tolist()}")
    exit()

numeric_cols = ['Params', 'Theoretical_FLOPS', 'Actual_FLOPS',
                'Activation_Size', 'Theoretical_Latency', 'Actual_Latency']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n--- Data Overview After Cleaning ---")
df.info()
print("\nData sample (first 5 rows):\n", df.head())
print("DataFrame shape:", df.shape)


# --- Visualization Functions ---

# 1. 3D Bubble Chart
print("\n--- Generating 1. 3D Bubble Chart ---")
fig1_data_cols = ['Theoretical_FLOPS', 'Actual_Latency', 'Activation_Size', 'Params', 'Input_Channels', 'Network']
fig1_data = df.dropna(subset=[col for col in fig1_data_cols if col in df.columns and col not in ['Input_Channels', 'Network']]) # Keep string cols
print(f"Shape for 3D Bubble Chart: {fig1_data.shape}")
if not fig1_data.empty:
    fig1 = go.Figure()
    # Ensure Input_Channels is treated as categorical for coloring
    fig1_data['Input_Channels_CatCode'] = fig1_data['Input_Channels'].astype('category').cat.codes
    for network_name in fig1_data['Network'].unique():
        subset = fig1_data[fig1_data['Network'] == network_name]
        fig1.add_trace(go.Scatter3d(
            x=subset['Theoretical_FLOPS'], y=subset['Actual_Latency'], z=subset['Activation_Size'],
            mode='markers', name=network_name,
            marker=dict(
                size=subset['Params'], sizemode='diameter',
                sizeref=max(1, fig1_data['Params'].max(skipna=True) / 30.0) if not fig1_data['Params'].empty and fig1_data['Params'].max(skipna=True) > 0 else 5,
                opacity=0.7, color=subset['Input_Channels_CatCode'], # Use categorical codes
                colorscale=px.colors.sequential.Viridis, colorbar_title_text='Input Channels (Category)'
            ),
            text=[f"Network: {net}<br>Input: {ic}<br>Params: {p:.2f}M<br>TFLOPs: {tf:.2f}<br>ALat: {al:.2f}ms<br>ActSize: {ac:.2f}MB"
                  for net, ic, p, tf, al, ac in zip(subset['Network'], subset['Input_Channels'], subset['Params'], subset['Theoretical_FLOPS'], subset['Actual_Latency'], subset['Activation_Size'])],
            hoverinfo='text'
        ))
    fig1.update_layout(
        scene=dict(xaxis_title='Theo FLOPS (GFLOPs)', yaxis_title='Actual Latency (ms)', zaxis_title='Activation Size (MB)'),
        title='1. 3D Performance Bubble Chart', legend_title_text='Network'
    )
    fig1.show()
    print("3D Bubble Chart generated.")
else:
    print("Skipping 3D Bubble Chart: Insufficient data.")

# 2. Pareto Frontier Plot
print("\n--- Generating 2. Pareto Frontier Plot ---")
fig2_data_cols = ['Theoretical_FLOPS', 'Actual_FLOPS', 'Network']
fig2_data = df.dropna(subset=[col for col in fig2_data_cols if col in df.columns])
print(f"Shape for Pareto Plot: {fig2_data.shape}")
if not fig2_data.empty:
    plt.figure(figsize=(12, 7))
    unique_networks_pareto = fig2_data['Network'].unique()
    colors_pareto = cm.viridis(np.linspace(0, 1, len(unique_networks_pareto)))
    for i, network_name in enumerate(unique_networks_pareto):
        subset = fig2_data[fig2_data['Network'] == network_name].copy()
        if subset.empty: continue
        plt.scatter(subset['Theoretical_FLOPS'], subset['Actual_FLOPS'], label=network_name, color=colors_pareto[i], alpha=0.7, s=50)
        sorted_subset = subset.sort_values(by=['Theoretical_FLOPS', 'Actual_FLOPS'], ascending=[True, False])
        pareto_front = []
        max_actual_flops_so_far = -np.inf
        for _, row in sorted_subset.iterrows(): # Simplified Pareto logic
            if row['Actual_FLOPS'] > max_actual_flops_so_far:
                pareto_front.append((row['Theoretical_FLOPS'], row['Actual_FLOPS']))
                max_actual_flops_so_far = row['Actual_FLOPS']
            elif row['Actual_FLOPS'] == max_actual_flops_so_far: # Handle same Y, prefer lower X
                 if pareto_front and row['Theoretical_FLOPS'] < pareto_front[-1][0]:
                    pareto_front.pop()
                    pareto_front.append((row['Theoretical_FLOPS'], row['Actual_FLOPS']))

        if pareto_front:
            pareto_front_np = np.array(sorted(list(set(pareto_front)), key=lambda x: x[0]))
            if pareto_front_np.ndim == 2 and pareto_front_np.shape[0] > 1 : # Check if plottable
                 plt.plot(pareto_front_np[:,0], pareto_front_np[:,1], '--', color=colors_pareto[i], linewidth=2)
    plt.xlabel('Theoretical FLOPS (GFLOPs)'); plt.ylabel('Actual FLOPS (GFLOPs)')
    plt.title('2. Actual FLOPS vs. Theoretical FLOPS (Illustrative Frontier)')
    plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.show()
    print("Pareto Frontier Plot generated.")
else:
    print("Skipping Pareto Plot: Insufficient data.")


# 3. Parallel Coordinates Plot
print("\n--- Generating 3. Parallel Coordinates Plot ---")
metrics_pcp_base = ['Params', 'Theoretical_FLOPS', 'Actual_FLOPS',
                    'Activation_Size', 'Theoretical_Latency', 'Actual_Latency']
# Ensure all listed metrics are actually in df.columns
metrics_pcp_present = [m for m in metrics_pcp_base if m in df.columns]
df_pcp = df.dropna(subset=metrics_pcp_present).copy() # Use .copy()
print(f"Shape for Parallel Coordinates (after dropna on numerics): {df_pcp.shape}")

if not df_pcp.empty and 'Network' in df_pcp.columns and 'Input_Channels' in df_pcp.columns and len(metrics_pcp_present) > 1:
    # Create a numerical representation of 'Input_Channels' for coloring
    df_pcp['Input_Channels_Code'] = df_pcp['Input_Channels'].astype('category').cat.codes
    
    dimensions_for_plot = metrics_pcp_present # These are the axes for the plot

    fig3 = px.parallel_coordinates(
        df_pcp,
        color='Input_Channels_Code',  # Use the numerical codes for color
        dimensions=dimensions_for_plot,
        labels={col: col.replace('_', ' ') for col in dimensions_for_plot}, # Nicer labels
        color_continuous_scale=px.colors.sequential.Viridis, # Use a continuous scale for the codes
        title='3. Multi-Dimensional Performance Profile (Colored by Input Channel Category)'
    )
    # Update color bar title if needed
    fig3.update_layout(coloraxis_colorbar=dict(
        title="Input Channel<br>(Category Code)",
        tickvals=df_pcp['Input_Channels_Code'].unique(), # Show ticks for unique codes
        ticktext=df_pcp.drop_duplicates(subset=['Input_Channels_Code']).sort_values('Input_Channels_Code')['Input_Channels'].tolist() # Map codes back to original strings for tick labels
    ))
    fig3.show()
    print("Parallel Coordinates Plot generated.")
else:
    print("Skipping Parallel Coordinates Plot: Insufficient data, missing 'Network'/'Input_Channels', or too few metrics.")


# 4. Radar Chart
print("\n--- Generating 4. Radar Chart ---")
radar_metrics = ['Params', 'Theoretical_FLOPS', 'Actual_FLOPS',
                 'Activation_Size', 'Theoretical_Latency', 'Actual_Latency']
radar_metrics_present = [m for m in radar_metrics if m in df.columns]
if 'Network' in df.columns and radar_metrics_present:
    df_radar_mean = df.groupby('Network')[radar_metrics_present].mean(numeric_only=True).reset_index()
    df_radar_mean.dropna(subset=radar_metrics_present, inplace=True)
    print(f"Shape for Radar (after mean & dropna): {df_radar_mean.shape}")
    if not df_radar_mean.empty and len(df_radar_mean) >= 1:
        df_radar_normalized = df_radar_mean.copy()
        for metric in radar_metrics_present: # Normalize
            min_val = df_radar_normalized[metric].min(); max_val = df_radar_normalized[metric].max()
            df_radar_normalized[metric] = (df_radar_normalized[metric] - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-9 else 0.5
        fig4 = go.Figure()
        for _, row in df_radar_normalized.iterrows():
            values = row[radar_metrics_present].tolist()
            fig4.add_trace(go.Scatterpolar(
                r=values + [values[0]], theta=radar_metrics_present + [radar_metrics_present[0]],
                fill='toself', name=row['Network'], hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
            ))
        fig4.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                           title='4. Normalized Average Network Performance (Radar Chart)', legend_title_text='Network')
        fig4.show()
        print("Radar Chart generated.")
    else:
        print("Skipping Radar Chart: No data after processing or too few networks.")
else:
    print("Skipping Radar Chart: 'Network' or metrics missing.")

# 5. Ridgeline Plot
print("\n--- Generating 5. Ridgeline Plot ---")
if 'Actual_Latency' in df.columns and 'Network' in df.columns:
    df_ridgeline = df.dropna(subset=['Actual_Latency', 'Network'])
    unique_networks_ridgeline = df_ridgeline['Network'].nunique()
    print(f"Shape for Ridgeline: {df_ridgeline.shape}, Unique Networks: {unique_networks_ridgeline}")
    if not df_ridgeline.empty and unique_networks_ridgeline > 1:
        plt.figure(figsize=(12, max(6, unique_networks_ridgeline * 0.8)))
        df_ridgeline_joy = df_ridgeline.copy()
        df_ridgeline_joy['Network'] = df_ridgeline_joy['Network'].astype(str)
        joypy.joyplot(data=df_ridgeline_joy[['Actual_Latency', 'Network']], by='Network', column='Actual_Latency',
                colormap=cm.plasma, overlap=1.2, linewidth=1, alpha=0.7, legend=False)
        plt.xlabel("Actual Latency (ms)"); plt.ylabel("Network Density")
        plt.title("5. Actual Latency Distribution Across Networks", pad=20); plt.show()
        print("Ridgeline Plot generated.")
    else:
        print("Skipping Ridgeline: Insufficient data or < 2 unique networks.")
else:
    print("Skipping Ridgeline: 'Actual_Latency' or 'Network' missing.")

# 6. Sankey Diagram
print("\n--- Generating 6. Sankey Diagram ---")
sankey_cols_req = ['Theoretical_FLOPS', 'Actual_FLOPS', 'Network', 'Input_Channels']
if all(col in df.columns for col in sankey_cols_req):
    df_sankey_prep = df.dropna(subset=sankey_cols_req).copy()
    df_sankey_prep['FLOPS_Difference'] = df_sankey_prep['Theoretical_FLOPS'] - df_sankey_prep['Actual_FLOPS']
    df_sankey_prep['FLOPS_Difference_Abs'] = df_sankey_prep['FLOPS_Difference'].abs()
    df_sankey = df_sankey_prep[df_sankey_prep['FLOPS_Difference_Abs'] > 1e-3].copy()
    print(f"Shape for Sankey: {df_sankey.shape}")
    if not df_sankey.empty:
        source_labels = df_sankey['Network'].astype(str).unique().tolist()
        target_labels = df_sankey['Input_Channels'].astype(str).unique().tolist()
        all_labels = source_labels + target_labels
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        df_sankey['source_idx'] = df_sankey['Network'].astype(str).map(label_to_idx)
        df_sankey['target_idx'] = df_sankey['Input_Channels'].astype(str).map(label_to_idx) # Map to combined index
        df_sankey.dropna(subset=['source_idx', 'target_idx'], inplace=True)
        df_sankey['target_idx'] = df_sankey['target_idx'].astype(int)
        if not df_sankey.empty:
            link_colors_sankey = ['rgba(255,0,0,0.4)' if diff > 0 else 'rgba(0,255,0,0.4)' for diff in df_sankey['FLOPS_Difference']]
            link_labels_sankey = [f"Diff: {diff:.2f} GFLOPs" for diff in df_sankey['FLOPS_Difference']]
            fig6 = go.Figure(data=[go.Sankey(
                node=dict(pad=20, thickness=25, line=dict(color="black", width=0.5), label=all_labels, color="blue"),
                link=dict(source=df_sankey['source_idx'].tolist(), target=df_sankey['target_idx'].tolist(),
                          value=df_sankey['FLOPS_Difference_Abs'].tolist(), label=link_labels_sankey,
                          color=link_colors_sankey, hovertemplate='Flow: %{value:.2f} GFLOPs<br>%{label}<extra></extra>')
            )])
            fig6.update_layout(title_text="6. Sankey: FLOPS Gap by Network & Input Channel", font_size=10)
            fig6.show()
            print("Sankey Diagram generated.")
        else:
            print("Skipping Sankey: No valid links.")
    else:
        print("Skipping Sankey: No data after filtering.")
else:
    print("Skipping Sankey: Required columns missing.")

print("\n--- Script finished ---")

#Contains Orignal Different types of plots made during Iteration1, for eg the Sankey Diagram, Radar Chart, etc.