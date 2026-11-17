
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np

import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from pathlib import Path


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
