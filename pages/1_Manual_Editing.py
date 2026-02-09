"""
SAR Water Detection - Manual Editing Page
==========================================

Streamlit page with drawable canvas for manual mask editing.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


def create_mask_image(mask, size=(512, 512)):
    """Convert binary mask to RGBA image for canvas background."""
    h, w = mask.shape
    
    # Create RGBA image
    img = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Water = blue with transparency
    img[mask, 0] = 30   # R
    img[mask, 1] = 144  # G
    img[mask, 2] = 255  # B
    img[mask, 3] = 180  # Alpha
    
    # Land = transparent
    img[~mask, 3] = 0
    
    pil_img = Image.fromarray(img, 'RGBA')
    
    # Resize to standard canvas size
    pil_img = pil_img.resize(size, Image.NEAREST)
    
    return pil_img


def canvas_to_mask(canvas_result, original_shape, mode='add'):
    """
    Convert canvas drawing to mask updates.
    
    Args:
        canvas_result: Result from st_canvas
        original_shape: Original mask shape (h, w)
        mode: 'add' or 'erase'
    
    Returns:
        mask_update: Boolean mask of changed pixels
    """
    if canvas_result.image_data is None:
        return None
    
    # Get the drawn image
    drawn = canvas_result.image_data
    
    # Convert to grayscale alpha
    alpha = drawn[:, :, 3]
    
    # Resize to original shape
    alpha_img = Image.fromarray(alpha)
    alpha_resized = alpha_img.resize((original_shape[1], original_shape[0]), Image.NEAREST)
    alpha_array = np.array(alpha_resized)
    
    # Create mask from drawings
    mask_update = alpha_array > 50
    
    return mask_update


def manual_editing_page():
    """Manual editing page with drawable canvas."""
    
    st.header("🎨 Manual Mask Editing")
    
    if not CANVAS_AVAILABLE:
        st.error("streamlit-drawable-canvas not installed. Run: `pip install streamlit-drawable-canvas`")
        return
    
    # Check if we have a composite mask to edit
    if 'composite_mask' not in st.session_state or st.session_state.composite_mask is None:
        st.warning("No mask to edit. Generate a composite first from the main page.")
        return
    
    mask = st.session_state.composite_mask
    chip_data = st.session_state.chip_data
    
    if chip_data is None:
        st.error("No chip data loaded.")
        return
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edit_mode = st.radio("Mode", ['✏️ Add Water', '🧹 Erase Water'])
    
    with col2:
        brush_size = st.slider("Brush Size", 5, 50, 20)
    
    with col3:
        st.metric("Current Water %", f"{mask.sum() / mask.size * 100:.2f}")
    
    # Create background image
    bg_image = create_mask_image(mask)
    
    # Drawing color
    if 'Add' in edit_mode:
        stroke_color = "#1E90FF"  # Blue for water
        drawing_mode = "freedraw"
    else:
        stroke_color = "#FF0000"  # Red for erase
        drawing_mode = "freedraw"
    
    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(30, 144, 255, 0.5)",
        stroke_width=brush_size,
        stroke_color=stroke_color,
        background_image=bg_image,
        update_streamlit=True,
        height=512,
        width=512,
        drawing_mode=drawing_mode,
        key="canvas"
    )
    
    # Apply changes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Apply Changes", use_container_width=True):
            if canvas_result.image_data is not None:
                update = canvas_to_mask(canvas_result, mask.shape)
                
                if update is not None:
                    if 'Add' in edit_mode:
                        st.session_state.composite_mask = mask | update
                    else:
                        st.session_state.composite_mask = mask & ~update
                    
                    st.success("Changes applied!")
                    st.rerun()
    
    with col2:
        if st.button("↩️ Undo All", use_container_width=True):
            if 'original_composite' in st.session_state:
                st.session_state.composite_mask = st.session_state.original_composite.copy()
                st.rerun()
    
    with col3:
        if st.button("💾 Save & Close", use_container_width=True):
            st.success("Mask saved!")


def polygon_lasso_page():
    """Polygon lasso editing page."""
    
    st.header("🔷 Polygon Lasso Tool")
    
    if not CANVAS_AVAILABLE:
        st.error("streamlit-drawable-canvas not installed.")
        return
    
    if 'composite_mask' not in st.session_state or st.session_state.composite_mask is None:
        st.warning("No mask to edit.")
        return
    
    mask = st.session_state.composite_mask
    
    # Mode selection
    mode = st.radio("Action", ['Fill Polygon (Add Water)', 'Clear Polygon (Remove Water)'])
    
    # Create background
    bg_image = create_mask_image(mask)
    
    # Canvas in polygon mode
    canvas_result = st_canvas(
        fill_color="rgba(30, 144, 255, 0.3)" if 'Fill' in mode else "rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFFF00",
        background_image=bg_image,
        update_streamlit=True,
        height=512,
        width=512,
        drawing_mode="polygon",
        key="polygon_canvas"
    )
    
    if st.button("Apply Polygon", use_container_width=True):
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            
            for obj in objects:
                if obj["type"] == "path":
                    # Extract polygon points
                    path = obj.get("path", [])
                    points = []
                    for cmd in path:
                        if len(cmd) >= 3 and cmd[0] in ['M', 'L']:
                            points.append((cmd[1], cmd[2]))
                    
                    if len(points) >= 3:
                        # Create polygon mask
                        from PIL import ImageDraw
                        poly_img = Image.new('L', (512, 512), 0)
                        draw = ImageDraw.Draw(poly_img)
                        draw.polygon(points, fill=255)
                        
                        # Resize to original
                        poly_resized = poly_img.resize((mask.shape[1], mask.shape[0]), Image.NEAREST)
                        poly_mask = np.array(poly_resized) > 128
                        
                        if 'Fill' in mode:
                            st.session_state.composite_mask = mask | poly_mask
                        else:
                            st.session_state.composite_mask = mask & ~poly_mask
            
            st.success("Polygon applied!")
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="Manual Editing", layout="wide")
    
    tab1, tab2 = st.tabs(["🎨 Paintbrush", "🔷 Polygon Lasso"])
    
    with tab1:
        manual_editing_page()
    
    with tab2:
        polygon_lasso_page()
