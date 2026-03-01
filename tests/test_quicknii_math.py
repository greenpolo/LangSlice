import numpy as np

def generate_quicknii_anchors(
    atlas_y: float,               # AP position (sagittal depth in Allen) -> This maps to Allen Y or Allen Z depending on axis
    rotation_deg: float, 
    translate_x: float,           # 2D translation 
    translate_y: float,           # 2D translation
    scale_x: float,               # Image to atlas scale
    scale_y: float,               # Image to atlas scale
    image_width: int,
    image_height: int,
    atlas="AMBA"
):
    """
    Generate [ox, oy, oz, ux, uy, uz, vx, vy, vz] QuickNII anchoring.
    Assuming Coronal sections (Constant AP axis).
    
    In ABBA/QuickNII for Allen Mouse Brain:
    X = Left-Right
    Y = Superior-Inferior (Dorso-Ventral)
    Z = Anterior-Posterior
    
    Let's test this mapping.
    """
    
    # 1. Base 2D corners of the image (in slice pixels)
    # Origin: [0, 0]
    # Top-Right: [image_width, 0]
    # Bottom-Left: [0, image_height]
    
    # 2. Apply Scale
    o_2d = np.array([0.0, 0.0])
    u_2d = np.array([image_width * scale_x, 0.0])
    v_2d = np.array([0.0, image_height * scale_y])
    
    # 3. Apply Rotation
    theta = np.radians(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_matrix = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    # We rotate the vectors, not the points (since u and v are vectors from origin)
    u_2d = np.dot(rot_matrix, u_2d)
    v_2d = np.dot(rot_matrix, v_2d)
    
    # 4. Apply Translation to the origin
    o_2d += np.array([translate_x, translate_y])
    
    # 5. Map to 3D Atlas Coordinates (Allen CCF)
    # If cutting Coronal slices, AP (Z-axis) is constant.
    # U vector is primarily along X (Left-Right)
    # V vector is primarily along Y (Superior-Inferior)
    
    ox = o_2d[0]
    oy = o_2d[1]
    oz = atlas_y 
    
    ux = u_2d[0]
    uy = u_2d[1]
    uz = 0.0 # No Z component for pure coronal slices
    
    vx = v_2d[0]
    vy = v_2d[1]
    vz = 0.0 # No Z component for pure coronal slices
    
    return [ox, oy, oz, ux, uy, uz, vx, vy, vz]

print(generate_quicknii_anchors(
    atlas_y=250.0,
    rotation_deg=5.0,
    translate_x=10.0,
    translate_y=-5.0,
    scale_x=0.025, # say 25um per pixel
    scale_y=0.025,
    image_width=1000,
    image_height=800
))
