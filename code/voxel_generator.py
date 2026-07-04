import plotly.graph_objects as go

def generate_3d_voxel(scene, grid_size):

    # 1. Munduaren koordenatuen limiteak
    limits_x = (-4, 4)
    limits_y = (-4, 4)
    limits_z = (0.0, 1.4)

    x_range = np.linspace(limits_x[0], limits_x[1], grid_size)
    y_range = np.linspace(limits_y[0], limits_y[1], grid_size)
    z_range = np.linspace(limits_z[0], limits_z[1], grid_size)

    # 2. `grid_size` tamaina duen grid-aren (X, Y, Z) guztiak gordetzen dira
    gx, gy, gz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    # 3. Nola jakin puntu bat objektu baten parte den?
    for obj in scene['objects']:

        # Objektuaren zentroa
        xc, yc, zc = obj['3d_coords']
        r = 0.35 if obj['size'] == 'small' else 0.7
        theta = np.radians(obj.get('rotation', 0))

        # Zein distantziatara dago puntu bakoitza objektuaren zentrora?
        dx, dy, dz = gx - xc, gy - yc, gz - zc

        if obj['shape'] == "sphere":
            mask = (dx**2 + dy**2 + dz**2) <= r**2
        elif obj['shape'] == "cube":
            dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
            dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)
            mask = (np.abs(dx_rot) <= r) & (np.abs(dy_rot) <= r) & (np.abs(dz) <= r)
        elif obj['shape'] == "cylinder":
            mask = (dx**2 + dy**2 <= r**2) & (np.abs(dz) <= r)

        voxels[mask] = 1

    return voxels


def visualize_grid(voxel_grid):

    limits_x = (-4, 4)
    limits_y = (-4, 4)
    limits_z = (0.0, 1.4)

    grid_size = voxel_grid.shape[0]

    step_x = (limits_x[1] - limits_x[0]) / grid_size
    step_y = (limits_y[1] - limits_y[0]) / grid_size
    step_z = (limits_z[1] - limits_z[0]) / grid_size

    x_idx, y_idx, z_idx = np.where(voxel_grid > 0)

    x_real = limits_x[0] + (x_idx + 0.5) * step_x
    y_real = limits_y[0] + (y_idx + 0.5) * step_y
    z_real = limits_z[0] + (z_idx + 0.5) * step_z

    fig = go.Figure(data=[go.Scatter3d(
        x=x_real, y=y_real, z=z_real,
        mode='markers', marker=dict(size=5, symbol='square', color='blue', opacity=1.0)
    )])


    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=limits_x),
            yaxis=dict(title='Y', range=limits_y),
            zaxis=dict(title='Z', range=limits_z),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.2)
        ),
        margin=dict(r=0, l=0, b=0, t=0)
    )

    fig.show()
