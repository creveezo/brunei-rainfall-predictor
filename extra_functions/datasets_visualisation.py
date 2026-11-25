
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np

import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from pathlib import Path
import time


# ==========================================
# Configuration
# ==========================================

MSWEP_FILE = "data/mswep_2024.nc"
SST_FILE = "data/sst_2024.nc"

WORLD_FILE = "data/world_countries.geojson"


def prepare_dataset(ds, variable):
    """
    Normalize coordinate names and longitude
    for geographic visualization.
    """

    da = ds[variable]

    # Detect coordinates
    aliases = {
        "latitude": ["lat", "latitude", "y"],
        "longitude": ["lon", "longitude", "x"],
    }

    rename = {}

    for target, candidates in aliases.items():
        for name in candidates:
            if name in da.coords or name in da.dims:
                if name != target:
                    rename[name] = target
                break
        else:
            raise ValueError(f"Cannot find {target} coordinate")

    da = da.rename(rename)

    if "time" not in da.dims:
        raise ValueError("Variable has no time dimension")

    # Convert longitude from 0..360 to -180..180
    lon = ((da.longitude + 180) % 360) - 180

    da = da.assign_coords(longitude=lon)
    da = da.sortby("longitude")
    da = da.sortby("latitude")

    return da


def load_world():
    return gpd.read_file(WORLD_FILE)


def get_boundary_coordinates(world):
    """Convert GeoPandas boundaries to Plotly coordinates."""

    xs, ys = [], []

    for geometry in world.geometry.boundary:

        if geometry is None:
            continue

        lines = (
            list(geometry.geoms)
            if hasattr(geometry, "geoms")
            else [geometry]
        )

        for line in lines:
            if not hasattr(line, "coords"):
                continue

            coords = list(line.coords)

            xs.extend([p[0] for p in coords])
            ys.extend([p[1] for p in coords])

            xs.append(None)
            ys.append(None)

    return xs, ys


def add_world_boundaries(fig, world, row=1, col=1):
    xs, ys = get_boundary_coordinates(world)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#555555", width=0.6),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def create_map(
    da,
    month_index,
    world,
    title,
    local_region=False,
    colorscale="Viridis",
):
    """
    Create an interactive monthly geographic map.
    local_region=True:
        Show regional detail only.
    local_region=False:
        Show a global data map.
    """
    month_data = da.isel(time=month_index)

    lat = da.latitude.values
    lon = da.longitude.values
    values = month_data.values

    # Remove extra singleton dimensions if present
    values = np.squeeze(values)

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2D monthly data, got {values.shape}"
        )

    world_x, world_y = get_boundary_coordinates(world)

    if local_region:

        fig = go.Figure()

        # Data map
        fig.add_trace(
            go.Heatmap(
                x=lon,
                y=lat,
                z=values,
                colorscale=colorscale,
                colorbar=dict(title="mm/month"),
                hovertemplate=(
                    "Longitude: %{x:.2f}°<br>"
                    "Latitude: %{y:.2f}°<br>"
                    "Value: %{z:.3f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=world_x,
                y=world_y,
                mode="lines",
                line=dict(color="#777777", width=0.7),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Regional zoom
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

        lon_pad = max((lon_max - lon_min) * 0.15, 0.15)
        lat_pad = max((lat_max - lat_min) * 0.15, 0.15)

        fig.update_xaxes(
            range=[lon_min - lon_pad, lon_max + lon_pad],
            title="Longitude",
        )

        fig.update_yaxes(
            range=[lat_min - lat_pad, lat_max + lat_pad],
            title="Latitude",
            scaleanchor="x",
        )

    else:

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=lon,
                y=lat,
                z=values,
                colorscale=colorscale,
                colorbar=dict(title=da.attrs.get("units", "")),
                hovertemplate=(
                    "Longitude: %{x:.2f}°<br>"
                    "Latitude: %{y:.2f}°<br>"
                    "Value: %{z:.3f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=world_x,
                y=world_y,
                mode="lines",
                line=dict(color="#777777", width=0.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(
            range=[-180, 180],
            title="Longitude",
        )

        fig.update_yaxes(
            range=[-85, 85],
            title="Latitude",
            scaleanchor="x",
        )

    fig.update_layout(
        title=title,
        height=550,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


# ==========================================
# Cached data loading
# ==========================================

@st.cache_resource
def load_netcdf(path):

    ds = xr.open_dataset(path)
    ds.load()

    return ds


@st.cache_resource
def load_world():

    import geopandas as gpd

    return gpd.read_file(WORLD_FILE)


# ==========================================
# Dataset visualization
# ==========================================

@st.fragment
def render_dataset_panel(
    title,
    path,
    variable,
    key,
    local_region=False,
    colorscale="Viridis",
):

    ds = load_netcdf(path)

    da = prepare_dataset(ds, variable)

    # Month slider
    month = st.slider(
        "Select month",
        min_value=1,
        max_value=12,
        value=1,
        key=f"{key}_month",
    )

    month_index = month - 1

    date = pd.Timestamp(da.time.values[month_index])

    st.markdown(f"**{date.strftime('%B %Y')}**")

    world = load_world()

    fig = create_map(
        da=da,
        month_index=month_index,
        world=world,
        title=title,
        local_region=local_region,
        colorscale=colorscale,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{key}_plot",
    )


# ==========================================
# Main section
# ==========================================

def show_dataset_explorer():

    col1, col2 = st.columns(2, gap="large")

    with col1:

        render_dataset_panel(
            title="Monthly Precipitation",
            path=MSWEP_FILE,
            variable="precipitation",
            key="mswep",
            local_region=True,
            colorscale="Blues",

        )


    with col2:
        render_dataset_panel(
            title="Global Sea Surface Temperature",
            path=SST_FILE,
            variable="sst",  # Replace with actual variable name
            key="sst",
            local_region=False,
            colorscale="RdBu_r",
        )


def plot_explained_variance(
    csv_path,
    selected_modes,
    title="EOF Explained Variance",
):
    """
    Plot individual and cumulative explained variance.

    selected_modes: number of leading EOF modes selected for analysis.
    """

    df = pd.read_csv(csv_path)

    fig = go.Figure()

    # Individual explained variance
    fig.add_trace(
        go.Scatter(
            x=df["mode"],
            y=df["explained_variance_percent"],
            mode="lines+markers",
            name="Explained variance",
            line=dict(
                color="#2878B5",
                width=2.5,
                dash="solid",
            ),
            marker=dict(size=7),
            hovertemplate=(
                "EOF%{x}<br>"
                "Explained variance: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    # Cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=df["mode"],
            y=df["cumulative_variance_percent"],
            mode="lines+markers",
            name="Cumulative explained variance",
            line=dict(
                color="#E69F00",
                width=2.5,
                dash="dash",
            ),
            marker=dict(size=7),
            hovertemplate=(
                "EOF%{x}<br>"
                "Cumulative variance: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    # Selected modes boundary
    fig.add_vline(
        x=selected_modes + 0.5,
        line=dict(
            color="red",
            width=2,
            dash="solid",
        ),
        annotation_text=f"Selected: EOF1–EOF{selected_modes}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="EOF Mode",
        yaxis_title="Explained Variance (%)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=30, t=90, b=50),
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=df["mode"],
        ticktext=[f"EOF{i}" for i in df["mode"]],
    )

    fig.update_yaxes(rangemode="tozero")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True},
    )




# ============================================================
# EOF AND PC ANALYSIS
# ============================================================

# NetCDF files used for EOF/PC visualization
MSWEP_EOF_FILE = "data/mswep_brunei_landmask_EOF_patterns_10_fullgrid.nc"
MSWEP_PC_TRAIN_FILE = "data/mswep_brunei_landmask_PC_train.nc"
MSWEP_PC_TEST_FILE = "data/mswep_brunei_landmask_PC_test.nc"
MSWEP_MASK_FILE = "data/brunei_landmask_used_for_EOF_on_mswep_grid.nc"

SST_EOF_PC_FILE = "data/sst_eof_svd_results.nc"


def create_eof_map(eof, world, title, colorbar_title="EOF loading"):
    """
    Plot one EOF spatial pattern with a symmetric diverging scale.
    """

    lat = eof.lat.values
    lon = eof.lon.values
    values = eof.values

    # Use a symmetric color scale centred at zero
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError(f"No valid EOF values found for {title}")

    limit = float(np.max(np.abs(finite_values)))

    if limit == 0:
        limit = 1.0

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=lon,
            y=lat,
            z=values,
            colorscale="RdBu_r",
            zmin=-limit,
            zmax=limit,
            zmid=0,
            colorbar=dict(title=colorbar_title),
            hovertemplate=(
                "Longitude: %{x:.2f}°<br>"
                "Latitude: %{y:.2f}°<br>"
                "EOF loading: %{z:.4f}"
                "<extra></extra>"
            ),
        )
    )

    # Add country boundaries
    world_x, world_y = get_boundary_coordinates(world)

    fig.add_trace(
        go.Scatter(
            x=world_x,
            y=world_y,
            mode="lines",
            line=dict(color="#555555", width=0.7),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Set geographic bounds with padding
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))

    lon_pad = max((lon_max - lon_min) * 0.05, 0.15)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.15)

    fig.update_layout(
        title=title,
        height=430,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
    )

    fig.update_xaxes(
        title="Longitude",
        range=[lon_min - lon_pad, lon_max + lon_pad],
    )

    fig.update_yaxes(
        title="Latitude",
        range=[lat_min - lat_pad, lat_max + lat_pad],
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def create_pc_plot(
    pc_train,
    pc_test,
    mode,
    title,
):
    """
    Plot training and test PC time series for one EOF mode.
    """

    train_dates = pd.to_datetime(pc_train.time.values)
    test_dates = pd.to_datetime(pc_test.time.values)

    train_values = pc_train.sel(mode=mode).values
    test_values = pc_test.sel(mode=mode).values

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_dates,
            y=train_values,
            mode="lines",
            name="Training period",
            line=dict(color="#2878B5", width=1.6),
            hovertemplate=(
                "%{x|%b %Y}<br>"
                "PC: %{y:.3f}"
                "<extra>Training</extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_values,
            mode="lines",
            name="Test period",
            line=dict(color="#E69F00", width=1.8),
            hovertemplate=(
                "%{x|%b %Y}<br>"
                "PC: %{y:.3f}"
                "<extra>Test</extra>"
            ),
        )
    )

    # Train/test boundary
    fig.add_vline(
        x=pd.Timestamp("2021-01-01"),
        line=dict(
            color="#555555",
            width=1.3,
            dash="dash",
        ),
        annotation_text="Test period",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Principal Component",
        height=430,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=45, r=20, t=80, b=45),
    )

    return fig


@st.fragment
def render_eof_pc_section(
    eof_data,
    pc_train,
    pc_test,
    world,
    key,
    colorbar_title="EOF loading",
):
    """
    Render one EOF map and its corresponding PC time series.
    A single selector controls both plots.
    """

    available_modes = eof_data.mode.values.tolist()

    selected_mode = st.selectbox(
        "Select EOF mode",
        options=available_modes,
        format_func=lambda x: f"EOF {x}",
        key=f"{key}_mode",
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        eof = eof_data.sel(mode=selected_mode)

        eof_fig = create_eof_map(
            eof=eof,
            world=world,
            title=f"EOF {selected_mode} — Spatial Pattern",
            colorbar_title=colorbar_title,
        )

        st.plotly_chart(
            eof_fig,
            use_container_width=True,
            key=f"{key}_eof_plot",
        )

    with col2:
        pc_fig = create_pc_plot(
            pc_train=pc_train,
            pc_test=pc_test,
            mode=selected_mode,
            title=f"PC {selected_mode} — Time Series",
        )

        st.plotly_chart(
            pc_fig,
            use_container_width=True,
            key=f"{key}_pc_plot",
        )


def show_eof_pc_analysis(SST=False):
    """
    Display EOF spatial patterns and PC time series
    for either MSWEP precipitation or Indian-Pacific SST.

    SST=True  -> SST analysis
    SST=False -> precipitation analysis
    """

    world = load_world()

    if SST:
        # --------------------------------------------------------
        # SST
        # --------------------------------------------------------

        sst_ds = load_netcdf(SST_EOF_PC_FILE)

        eof_data = sst_ds["EOF"]
        sst_mask = sst_ds["valid_ocean_mask"]

        # Mask land and invalid grid cells
        eof_data = eof_data.where(sst_mask == 1)

        # Convert longitudes from 0..360 to -180..180
        sst_lon = ((eof_data.lon + 180) % 360) - 180

        eof_data = (
            eof_data
            .assign_coords(lon=sst_lon)
            .sortby("lon")
        )

        pc_train = sst_ds["PC_train"].rename(
            {"time_train": "time"}
        )
        pc_test = sst_ds["PC_test"].rename(
            {"time_test": "time"}
        )

        key = "sst"
        colorbar_title = "EOF loading"

    else:
        # --------------------------------------------------------
        # MSWEP PRECIPITATION
        # --------------------------------------------------------

        mswep_eof_ds = load_netcdf(MSWEP_EOF_FILE)
        mswep_train_ds = load_netcdf(MSWEP_PC_TRAIN_FILE)
        mswep_test_ds = load_netcdf(MSWEP_PC_TEST_FILE)
        mswep_mask_ds = load_netcdf(MSWEP_MASK_FILE)

        # Keep only Brunei land-mask cells
        eof_data = mswep_eof_ds["EOF"].where(
            mswep_mask_ds["brunei_landmask"] == 1
        )

        pc_train = mswep_train_ds["PC"]
        pc_test = mswep_test_ds["PC"]

        key = "mswep"
        colorbar_title = "EOF loading"

    # ------------------------------------------------------------
    # RENDER EOF MAP AND PC TIME SERIES
    # ------------------------------------------------------------

    render_eof_pc_section(
        eof_data=eof_data,
        pc_train=pc_train,
        pc_test=pc_test,
        world=world,
        key=key,
        colorbar_title=colorbar_title,
    )


def show_correlation_analysis():
    """
    Display correlations between Indian-Pacific SST PCs
    and Brunei precipitation PCs using the training period.
    """

    st.subheader("Correlation Analysis")

    st.markdown("""
    Correlation analysis measures the strength and direction
    of linear relationships between SST and precipitation PCs.
    """)

    # Load PC datasets
    sst_ds = load_netcdf(SST_EOF_PC_FILE)
    mswep_train_ds = load_netcdf(MSWEP_PC_TRAIN_FILE)

    # SST training PCs: (time_train, mode) -> (time, mode)
    sst_pc = sst_ds["PC_train"].rename(
        {"time_train": "time"}
    )

    # Precipitation training PCs
    pr_pc = mswep_train_ds["PC"]

    # Align both datasets by dates
    sst_pc, pr_pc = xr.align(
        sst_pc,
        pr_pc,
        join="inner"
    )

    if sst_pc.sizes["time"] == 0:
        st.error("No overlapping dates found between SST and precipitation PCs.")
        return

    # Convert to DataFrames
    sst_df = sst_pc.to_pandas()
    pr_df = pr_pc.to_pandas()

    # Rename columns for clarity
    sst_df.columns = [f"SST PC {i}" for i in sst_df.columns]
    pr_df.columns = [f"Precipitation PC {i}" for i in pr_df.columns]

    # --------------------------------------------------------
    # 1. CORRELATION MATRIX
    # --------------------------------------------------------

    st.markdown("### Correlation Matrix")

    correlation_matrix = pd.DataFrame(
        index=sst_df.columns,
        columns=pr_df.columns,
        dtype=float
    )

    for sst_col in sst_df.columns:
        for pr_col in pr_df.columns:
            correlation_matrix.loc[sst_col, pr_col] = (
                sst_df[sst_col].corr(pr_df[pr_col])
            )

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            zmid=0,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            colorbar=dict(title="Pearson r"),
            hovertemplate=(
                "%{y}<br>"
                "%{x}<br>"
                "Correlation: %{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig_corr.update_layout(
        title="SST PCs vs Precipitation PCs",
        xaxis_title="Precipitation Principal Components",
        yaxis_title="SST Principal Components",
        height=550,
        template="plotly_white",
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
        key="pc_correlation_matrix",
    )

    # --------------------------------------------------------
    # 2. SCATTER PLOTS
    # --------------------------------------------------------

    st.markdown("### SST PCs vs Precipitation PC1")

    st.markdown("""
    Each scatter plot shows the relationship between one SST
    principal component and the first precipitation PC.
    The red line represents the least-squares linear regression.
    """)

    target = pr_df["Precipitation PC 1"]

    cols = st.columns(2, gap="large")

    for i, sst_col in enumerate(sst_df.columns[:6]):

        x = sst_df[sst_col].values
        y = target.values

        # Pearson correlation
        r = np.corrcoef(x, y)[0, 1]

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Observations",
                marker=dict(
                    size=6,
                    opacity=0.65,
                ),
                hovertemplate=(
                    "SST PC: %{x:.3f}<br>"
                    "Precipitation PC1: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Linear regression",
                line=dict(
                    color="red",
                    width=2,
                ),
            )
        )

        fig.update_layout(
            title=f"{sst_col} vs Precipitation PC 1<br>Pearson r = {r:.3f}",
            xaxis_title=sst_col,
            yaxis_title="Precipitation PC 1",
            height=350,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=20, t=75, b=40),
        )

        with cols[i % 2]:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"scatter_{i}",
            )

    st.caption(
        f"Training period: {sst_df.index.min():%Y-%m} — "
        f"{sst_df.index.max():%Y-%m} | "
        f"{len(sst_df)} observations"
    )
