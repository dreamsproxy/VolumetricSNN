import numpy as np
import plotly.graph_objects as go
from glob import glob
import os


def rotate_points(scatter_points, rotation_matrix):
    # Apply rotations to the scatter points
    rotated_points = []
    #print(rotation_matrix.shape)
    for point in scatter_points:
        rotated_point = np.dot(point, rotation_matrix.T)
        rotated_points.append(rotated_point)
    rotated_points = np.array(rotated_points)
    return rotated_points

def translate_points(scatter_points, translation_vector):
    translated_points = []
    for point in scatter_points:
        translated_point = point + translation_vector
        translated_points.append(translated_point)
    translated_points = np.array(translated_points)
    return translated_points

def normalize_to_range(values):
    min_val = min(values)
    max_val = max(values)
    normalized = [(val - min_val) / (max_val - min_val) for val in values]
    return np.array(normalized)

main_dir = "./dataset/Streets"
points_path = glob(f"{main_dir}/points/**")
points_3d = []

for p in points_path:
    points = np.loadtxt(p)
    img_id = os.path.basename(p).split("_")[0]
    # Get r and t mats
    rmat = np.loadtxt(f"{main_dir}/rmat/{img_id}_rmat.txt")
    tmat = np.loadtxt(f"{main_dir}/tmat/{img_id}_tmat.txt")
    # Apply transforms
    points[0] = normalize_to_range(points[0])
    points[1] = normalize_to_range(points[1])
    points[2] = normalize_to_range(points[2])
    points = translate_points(points, tmat)
    points = rotate_points(points, rmat)
    if len(points_3d) >= 1:
        points_3d[0] = np.vstack([points_3d[0], points])
    else:
        points_3d.append(points)

points_3d = points_3d[0]

points_3d = np.asarray(points_3d, dtype=np.float32)
points_3d[0] = normalize_to_range(points_3d[0])
points_3d[1] = normalize_to_range(points_3d[1])
points_3d[2] = normalize_to_range(points_3d[2])
#print(points_3d.shape)
#raise
#target_shape = (points_3d.shape[0] * points_3d.shape[1], points_3d.shape[2])
#points_3d = np.reshape(points_3d, target_shape)
#raise

fig = go.Figure(data=go.Scatter3d(
    x=[point[0] for point in points_3d],
    y=[point[1] for point in points_3d],
    z=[point[2] for point in points_3d],
    mode='markers',
    surfacecolor = "black",
    marker=dict(
        size=2,
        colorscale='gray',  # Choose the colorscale (you can change this)
        opacity=1.0
        #colorbar=dict(title='Normalized Num Neighbors')
    )
))
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z',
), title='Filtered Point Cloud with Color-Coded Normalized using number of neighbors in the radius of 0.15')

fig.show()