import numpy as np

def calculate_density(x, y, z):
    # Assuming each unit space is 10 cubed
    unit_size = 10

    # Determine the grid dimensions based on the range of coordinates
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    z_min, z_max = min(z), max(z)

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Calculate the number of grid cells in each dimension
    x_cells = int(np.ceil(x_range / unit_size))
    y_cells = int(np.ceil(y_range / unit_size))
    z_cells = int(np.ceil(z_range / unit_size))

    # Initialize a grid to count points in each cell
    grid = np.zeros((x_cells, y_cells, z_cells), dtype=int)

    # Populate the grid with point counts
    for point_x, point_y, point_z in zip(x, y, z):
        x_index = int((point_x - x_min) / unit_size)
        y_index = int((point_y - y_min) / unit_size)
        z_index = int((point_z - z_min) / unit_size)

        grid[x_index, y_index, z_index] += 1

    # Calculate the density for each cell (points per unit)
    density_per_unit = grid / unit_size**3

    return density_per_unit