"""
SAR Water Detection - Map View Page
=====================================

Geo-sync map with folium integration.
"""

import streamlit as st
import numpy as np

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


def get_chip_bounds(chip_data):
    """Extract bounds from chip data."""
    if chip_data is None:
        return None
    return chip_data.get('bounds')


def create_base_map(bounds, zoom=14):
    """Create folium map centered on chip bounds."""
    if bounds is None:
        return None
    
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None
    )
    
    # Add multiple tile layers
    folium.TileLayer(
        'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        name='Google Satellite',
        attr='Google'
    ).add_to(m)
    
    folium.TileLayer(
        'OpenStreetMap',
        name='OpenStreetMap'
    ).add_to(m)
    
    folium.TileLayer(
        'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        name='Google Hybrid',
        attr='Google'
    ).add_to(m)

    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        name='Esri World Imagery',
        attr='Esri'
    ).add_to(m)
    
    # Add chip boundary rectangle
    folium.Rectangle(
        bounds=[[miny, minx], [maxy, maxx]],
        color='red',
        weight=3,
        fill=True,
        fillColor='red',
        fillOpacity=0.1,
        popup=f"Chip Area: {(maxx-minx)*111:.2f} x {(maxy-miny)*111:.2f} km"
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m


def add_mask_overlay(m, mask, bounds, color='blue', opacity=0.5):
    """Add mask overlay to map."""
    if mask is None or bounds is None:
        return m
    
    minx, miny, maxx, maxy = bounds
    
    # Create image overlay
    # This is simplified - full implementation would create a proper GeoTIFF overlay
    
    # For now, just add a marker at center
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    
    water_pct = mask.sum() / mask.size * 100
    
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f"Water: {water_pct:.1f}%",
        icon=folium.Icon(color='blue', icon='tint')
    ).add_to(m)
    
    return m


def map_view_page():
    """Map view page with geo-sync."""
    
    st.header("🗺️ Geo-Sync Map View")
    
    if not FOLIUM_AVAILABLE:
        st.error("folium or streamlit-folium not installed. Run: `pip install folium streamlit-folium`")
        return
    
    # Check for chip data
    if 'chip_data' not in st.session_state or st.session_state.chip_data is None:
        st.warning("No chip data loaded. Go to main page and select a chip.")
        return
    
    chip_data = st.session_state.chip_data
    bounds = get_chip_bounds(chip_data)
    
    if bounds is None:
        st.error("No geographic bounds available for this chip.")
        return
    
    # Sidebar options
    with st.sidebar:
        st.header("Map Options")
        
        show_mask = st.checkbox("Show Water Mask Overlay", value=True)
        zoom_level = st.slider("Initial Zoom", 10, 18, 14)
    
    # Create map
    m = create_base_map(bounds, zoom=zoom_level)
    
    if m is None:
        st.error("Could not create map.")
        return
    
    # Add mask overlay if available
    if show_mask and 'composite_mask' in st.session_state:
        m = add_mask_overlay(m, st.session_state.composite_mask, bounds)
    
    # Display map
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st_folium(m, width=800, height=600)
    
    with col2:
        st.subheader("📍 Location Info")
        
        minx, miny, maxx, maxy = bounds
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2
        
        st.write(f"**Center:** {center_lat:.4f}°N, {center_lon:.4f}°E")
        st.write(f"**Width:** {(maxx-minx)*111:.2f} km")
        st.write(f"**Height:** {(maxy-miny)*111:.2f} km")
        
        # External links
        st.divider()
        st.subheader("🔗 External Links")
        
        google_url = f"https://www.google.com/maps/@{center_lat},{center_lon},15z/data=!3m1!1e3"
        st.markdown(f"[🗺️ Google Maps]({google_url})")
        
        osm_url = f"https://www.openstreetmap.org/#map=15/{center_lat}/{center_lon}"
        st.markdown(f"[🌍 OpenStreetMap]({osm_url})")
        
        bing_url = f"https://www.bing.com/maps?cp={center_lat}~{center_lon}&lvl=15&style=a"
        st.markdown(f"[🔷 Bing Maps]({bing_url})")
        
        # Sentinel Playground (Time Machine)
        sentinel_url = f"https://apps.sentinel-hub.com/sentinel-playground/?source=S2&lat={center_lat}&lng={center_lon}&zoom=14"
        st.markdown(f"[🛰️ Sentinel Time-Machine]({sentinel_url})")


def dual_view_page():
    """Dual view: SAR vs Optical."""
    
    st.header("👁️ Dual View Comparison")
    
    if not FOLIUM_AVAILABLE:
        st.error("folium not installed.")
        return
    
    if 'chip_data' not in st.session_state or st.session_state.chip_data is None:
        st.warning("No chip data loaded.")
        return
    
    chip_data = st.session_state.chip_data
    bounds = get_chip_bounds(chip_data)
    
    if bounds is None:
        st.error("No bounds available.")
        return
    
    # Create two maps side by side
    col1, col2 = st.columns(2)
    
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    
    with col1:
        st.subheader("SAR VH Band")
        # Show SAR image
        import matplotlib.pyplot as plt
        
        vh_data = chip_data.get('vh')
        if vh_data is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(vh_data, cmap='gray', vmin=-30, vmax=-10)
            ax.axis('off')
            ax.set_title(f"Sentinel-1 VH (min: {vh_data.min():.1f}, max: {vh_data.max():.1f} dB)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='dB')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error("VH band data not available in chip_data")
    
    with col2:
        st.subheader("Satellite View")
        # Show map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google'
        )
        
        folium.Rectangle(
            bounds=[[miny, minx], [maxy, maxx]],
            color='red',
            weight=2,
            fill=False
        ).add_to(m)
        
        st_folium(m, width=400, height=400)


if __name__ == "__main__":
    st.set_page_config(page_title="Map View", layout="wide")
    
    tab1, tab2 = st.tabs(["🗺️ Map View", "👁️ Dual View"])
    
    with tab1:
        map_view_page()
    
    with tab2:
        dual_view_page()
