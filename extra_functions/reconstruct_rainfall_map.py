
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# 1. LOAD GEOGRAPHICAL BOUNDARIES
# --------------------------------------------------

@st.cache_data
def load_boundaries():

    brunei = gpd.read_file("data/brunei.geojson")
    malaysia = gpd.read_file("data/malaysia.geojson")

    # GeoJSON is normally WGS84.
    # Convert projected data to longitude/latitude if necessary.
    for gdf in (brunei, malaysia):
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf.to_crs("EPSG:4326", inplace=True)

    return brunei, malaysia


# --------------------------------------------------
# 2. EXTRACT POLYGONS
# --------------------------------------------------

def iter_polygons(geometry):

    if geometry is None or geometry.is_empty:
        return

    if geometry.geom_type == "Polygon":
        yield geometry

    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms

    elif geometry.geom_type == "GeometryCollection":
        for geom in geometry.geoms:
            yield from iter_polygons(geom)


# --------------------------------------------------
# 3. DRAW LAND AS BACKGROUND SHAPES
# --------------------------------------------------

def add_land_background(fig, gdf):

    for geometry in gdf.geometry:

        for polygon in iter_polygons(geometry):

            coordinates = list(polygon.exterior.coords)

            # Convert polygon coordinates to SVG path
            path = (
                f"M {coordinates[0][0]},{coordinates[0][1]} "
                + " ".join(
                    f"L {lon},{lat}"
                    for lon, lat in coordinates[1:]
                )
                + " Z"
            )

            fig.add_shape(
                type="path",
                path=path,
                xref="x",
                yref="y",
                fillcolor="#E8E4D8",
                line=dict(
                    color="#A7A49B",
                    width=0.5
                ),
                layer="below"
            )


# --------------------------------------------------
# 4. DRAW COUNTRY BOUNDARIES
# --------------------------------------------------

def add_boundary_lines(fig, gdf, color, width):

    for geometry in gdf.geometry:

        for polygon in iter_polygons(geometry):

            coordinates = list(polygon.exterior.coords)

            lons = [point[0] for point in coordinates]
            lats = [point[1] for point in coordinates]

            fig.add_trace(
                go.Scatter(
                    x=lons,
                    y=lats,
                    mode="lines",
                    line=dict(
                        color=color,
                        width=width
                    ),
                    hoverinfo="skip",
                    showlegend=False
                )
            )


# --------------------------------------------------
# 5. MAIN MAP FUNCTION
# --------------------------------------------------

def plot_rainfall_map(anomaly, show_values=False):

    # Get coordinates and predicted anomaly values
    lons = anomaly["lon"].values
    lats = anomaly["lat"].values
    values = np.asarray(anomaly.values, dtype=float)

    # Check dimensions
    if values.shape != (len(lats), len(lons)):
        raise ValueError(
            "Anomaly dimensions do not match latitude and longitude coordinates."
        )

    # Color scale centered around zero
    finite_values = values[np.isfinite(values)]

    if len(finite_values) == 0:
        st.error("No valid anomaly values to display.")
        return

    max_abs = np.max(np.abs(finite_values))

    if max_abs == 0:
        max_abs = 1

    # Load geographical data
    brunei, malaysia = load_boundaries()

    fig = go.Figure()

    # --------------------------------------------------
    # A. LAND BACKGROUND
    # Shapes with layer='below' are placed underneath
    # the heatmap, regardless of trace ordering.
    # --------------------------------------------------

    add_land_background(fig, malaysia)
    add_land_background(fig, brunei)

    # --------------------------------------------------
    # B. RAINFALL ANOMALY HEATMAP
    # --------------------------------------------------

    fig.add_trace(
        go.Heatmap(
            x=lons,
            y=lats,
            z=values,
            colorscale="RdBu_r",
            zmin=-max_abs,
            zmax=max_abs,
            zmid=0,
            colorbar=dict(
                title="Rainfall anomaly",
                ticksuffix=" mm"
            ),
            hovertemplate=(
                "Longitude: %{x:.2f}°<br>"
                "Latitude: %{y:.2f}°<br>"
                "Anomaly: %{z:+.2f} mm"
                "<extra></extra>"
            ),
            connectgaps=False
        )
    )

    # --------------------------------------------------
    # C. NUMERICAL VALUES ON THE MAP
    # --------------------------------------------------

    if show_values:

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        text_labels = []

        for value in values.flatten():

            if not np.isfinite(value):
                text_labels.append("")
            elif value >= 0:
                text_labels.append(f"+{value:.1f}")
            else:
                text_labels.append(f"{value:.1f}")

        fig.add_trace(
            go.Scatter(
                x=lon_grid.flatten(),
                y=lat_grid.flatten(),
                mode="text",
                text=text_labels,
                textfont=dict(
                    size=10,
                    color="black"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # --------------------------------------------------
    # D. COUNTRY BOUNDARIES ON TOP
    # --------------------------------------------------

    add_boundary_lines(
        fig,
        malaysia,
        color="#777777",
        width=1
    )

    add_boundary_lines(
        fig,
        brunei,
        color="#222222",
        width=2
    )

    # --------------------------------------------------
    # E. MAP LAYOUT
    # --------------------------------------------------

    lon_margin = 0.25
    lat_margin = 0.20

    fig.update_layout(
        title="Reconstructed Rainfall Anomaly (EOF1)",

        xaxis_title="Longitude",
        yaxis_title="Latitude",

        template="plotly_white",
        height=650,

        plot_bgcolor="#DCEAF2",
        paper_bgcolor="white",

        margin=dict(
            l=50,
            r=50,
            t=70,
            b=50
        ),

        xaxis=dict(
            range=[
                float(np.min(lons) - lon_margin),
                float(np.max(lons) + lon_margin)
            ],
            showgrid=True,
            gridcolor="white",
            zeroline=False
        ),

        yaxis=dict(
            range=[
                float(np.min(lats) - lat_margin),
                float(np.max(lats) + lat_margin)
            ],
            showgrid=True,
            gridcolor="white",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        )
    )

    # --------------------------------------------------
    # F. DISPLAY IN STREAMLIT
    # --------------------------------------------------

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "rainfall_anomaly_map",
                "scale": 2
            }
        }
    )
