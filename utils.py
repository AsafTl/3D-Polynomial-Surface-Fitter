import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from skimage.measure import regionprops_table
 
def perform_dbscan_clustering(x, y, eps=50, min_samples=10):
    """
    Performs DBSCAN clustering on X, Y coordinates.
    
    """
    xy_coords = np.column_stack((x, y))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(xy_coords)
    labels = dbscan.labels_
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    return labels, core_samples_mask

def filter_data_by_cluster(labels):
    """
    Identifies indices of points belonging to the largest DBSCAN cluster.

    Args:
        labels (np.array): DBSCAN cluster labels.

    Returns:
        np.array: Indices of points in the largest cluster.
                   Returns None if no clusters found (all noise).
    """
    unique_labels = set(labels)
    cluster_counts = {}
    for label in unique_labels:
        if label != -1: # Ignore noise points
            cluster_counts[label] = np.sum(labels == label)

    if not cluster_counts: # No clusters found (only noise)
        print("Warning: No clusters found by DBSCAN. All points are noise.")
        return None

    largest_cluster_label = max(cluster_counts, key=cluster_counts.get)
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    print(f"Largest cluster label: {largest_cluster_label}, Size: {len(largest_cluster_indices)} points")
    return largest_cluster_indices # Return indices instead of filtered arrays



def fit_polynomial_surface(x, y, z, degree):
    """
    Fits a polynomial surface of a given degree to 3D points.

    Args:
        x (np.array): 1D array of X coordinates.
        y (np.array): 1D array of Y coordinates.
        z (np.array): 1D array of Z coordinates.
        degree (int): Degree of the polynomial.

    Returns:
        tuple: A tuple containing:
            - model (LinearRegression): The fitted linear regression model.
            - poly_features (PolynomialFeatures): The polynomial feature transformer.
            - z_predicted (np.array): Predicted Z values for the input X, Y.
    """
    # Create polynomial features from X and Y coordinates
    poly_features = PolynomialFeatures(degree=degree)
    xy_features = poly_features.fit_transform(np.column_stack((x, y)))

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(xy_features, z)

    # Predict Z values
    z_predicted = model.predict(xy_features)

    return model, poly_features, z_predicted, z - z_predicted

def predict_blob_surface(x, y, poly_features, model):
    """
    Fits a polynomial surface of a given degree to 3D points.

    Args:
        x (np.array): 1D array of X coordinates.
        y (np.array): 1D array of Y coordinates.
        z (np.array): 1D array of Z coordinates.
        poly_features : (PolynomialFeatures): The polynomial feature transformer.

    Returns:
        
        - z_predicted (np.array): Predicted Z values for the input X, Y.
    """
    
    xy_features = poly_features.transform(np.column_stack((x, y)))


    # Predict Z values
    z_predicted = model.predict(xy_features)

    return z_predicted

def evaluate_fit(z_true, z_predicted, n_features):
    """
    Evaluates the goodness of fit using RMSE and Adjusted R-squared.

    Args:
        z_true (np.array): True Z values.
        z_predicted (np.array): Predicted Z values.
        n_features (int): Number of features in the model (excluding intercept).

    Returns:
        tuple: A tuple containing RMSE and Adjusted R-squared.
    """
    rmse = np.sqrt(mean_squared_error(z_true, z_predicted))
    r_squared = r2_score(z_true, z_predicted)
    n_samples = len(z_true)
    adjusted_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)
    return rmse, adjusted_r_squared

def plot_residuals_histogram(residuals_normalized, degree, dataset_label=""):
    """
    Plots a histogram of the normalized residuals.
    (Modified to accept dataset label for title)
    """
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_normalized, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of Normalized Residuals (Degree {degree}) - {dataset_label}') # Dataset label in title
    plt.xlabel('Normalized Residuals (in std dev of Residuals)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def filter_outliers(x, y, z, residuals_normalized, threshold_std_dev=2.0):
    """
    Filters out outlier points based on normalized residuals.

    Args:
        x (np.array): 1D array of X coordinates.
        y (np.array): 1D array of Y coordinates.
        z (np.array): 1D array of Z coordinates.
        residuals_normalized (np.array): Normalized residuals.
        threshold_std_dev (float): Threshold in standard deviations for outlier detection.

    Returns:
        tuple: Tuple containing inlier arrays for x, y, z.
    """
    outlier_indices = np.where(np.abs(residuals_normalized) > threshold_std_dev)[0]
    inlier_indices = np.where(np.abs(residuals_normalized) <= threshold_std_dev)[0]

    inlier_x = x[inlier_indices]
    inlier_y = y[inlier_indices]
    inlier_z = z[inlier_indices]

    print(f"Number of outliers removed: {len(outlier_indices)}")
    print(f"Number of inliers remaining: {len(inlier_indices)}")

    return inlier_x, inlier_y, inlier_z, inlier_indices # Return inlier indices for potential use


def create_surface_volume_optimized(inlier_x, inlier_y, model_inliers, poly_features_inliers, z_size):
    """
    Creates a 3D NumPy array representing the polynomial surface within the convex hull (optimized).

    Args:
        inlier_x (np.array): X coordinates of inliers.
        inlier_y (np.array): Y coordinates of inliers.
        model_inliers (LinearRegression): Fitted linear regression model for inliers.
        poly_features_inliers (PolynomialFeatures): Polynomial feature transformer for inliers.
        z_max_data (float): Maximum Z value from the original data to determine volume height.

    Returns:
        np.array: 3D NumPy array (1024x1024xZ) representing the surface.
    """
    volume_size_xy = 1024
    surface_volume = np.zeros((volume_size_xy, volume_size_xy, z_size), dtype=np.uint8)

    #x_range = np.linspace(min(inlier_x)-0, max(inlier_x)+0, volume_size_xy)
    #y_range = np.linspace(min(inlier_y)-0, max(inlier_y)+0, volume_size_xy)
    x_range = np.linspace(0, 1023, volume_size_xy)
    y_range = np.linspace(0, 1023, volume_size_xy)
    grid_x, grid_y = np.meshgrid(x_range, y_range, indexing='xy') # 2D grids for x and y

    # --- 1. Vectorized Convex Hull Check ---
    points_2d = np.column_stack((inlier_x, inlier_y))
    hull = ConvexHull(points_2d)
    hull_path = Path(points_2d[hull.vertices])

    xy_grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel())) # Reshape grid to points
    inside_hull_mask = hull_path.contains_points(xy_grid_points).reshape(volume_size_xy, volume_size_xy) # Mask for points inside hull
    
    # --- 2. Vectorized Z Prediction (for points inside hull) ---
    xy_inside_hull = np.floor(xy_grid_points[inside_hull_mask.ravel()]).astype(int) # Get XY points inside hull
    if xy_inside_hull.size > 0: # Proceed only if there are points inside hull
        xy_features_inside_hull = poly_features_inliers.transform(xy_inside_hull)
        predicted_z_values = model_inliers.predict(xy_features_inside_hull)
        predicted_z_coords = np.floor(predicted_z_values).astype(int) # Voxel indices, floor to integer

        # --- 3. Vectorized Volume Population ---
        valid_z_indices = (predicted_z_coords >= 0) & (predicted_z_coords < z_size) # Valid Z voxel indices

        xy_indices_inside_hull = np.transpose(np.where(inside_hull_mask)) # Get 2D indices (row, col) for inside hull

        # Use advanced indexing to set voxels to 1 (vectorized)
        volume_x_indices = xy_indices_inside_hull[valid_z_indices, 1] # Column indices are X in 'xy' indexing
        volume_y_indices = xy_indices_inside_hull[valid_z_indices, 0] # Row indices are Y in 'xy' indexing
        volume_z_indices = predicted_z_coords[valid_z_indices]

        surface_volume[volume_y_indices, volume_x_indices, volume_z_indices] = 1


    return surface_volume




def visualize_surface_fit(surface_volume_3d, inlier_x, inlier_y, inlier_z):
    """
    Visualizes the surface fit using Max XZ, Max YZ, and Max XY projections, color-coded by XY proximity.
    """
    xz_projection_max = np.max(surface_volume_3d, axis=1)
    yz_projection_max = np.max(surface_volume_3d, axis=0)
    xy_projection_max = np.max(surface_volume_3d, axis=2)

    z_size = surface_volume_3d.shape[2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- XY Proximity Coloring ---
    xy_proximity_values_xy = inlier_y  # Use inlier_y as proximity for XY (example - you can choose X or combine)
    xy_proximity_values_xz = inlier_y  # Use inlier_y as proximity for XZ
    xy_proximity_values_yz = inlier_x  # Use inlier_x as proximity for YZ


    # XZ Projection (Max)
    ax_xz = axes[0]
    ax_xz.imshow(xz_projection_max.T, origin='lower', extent=[min(inlier_x), max(inlier_x), 0, z_size], aspect='auto', cmap='gray')
    scatter_xz = ax_xz.scatter(inlier_x, inlier_z, s=5, c=xy_proximity_values_xz, cmap='plasma', label='Inlier Data Points') # Color by XY proximity (Y-coordinate)
    ax_xz.set_xlabel('X (px)')
    ax_xz.set_ylabel('Z (px)')
    ax_xz.set_title('Max XZ Projection - Color-coded by Y-proximity') # Updated title
    ax_xz.legend()
    ax_xz.set_xlim([min(inlier_x) - 50, max(inlier_x) + 50])
    ax_xz.set_ylim([0, z_size])
    plt.colorbar(scatter_xz, ax=ax_xz, label='Y-Proximity') # Colorbar for XZ plot


    # YZ Projection (Max)
    ax_yz = axes[1]
    ax_yz.imshow(yz_projection_max.T, origin='lower', extent=[min(inlier_y), max(inlier_y), 0, z_size], aspect='auto', cmap='gray')
    scatter_yz = ax_yz.scatter(inlier_y, inlier_z, s=5, c=xy_proximity_values_yz, cmap='plasma', label='Inlier Data Points') # Color by XY proximity (X-coordinate)
    ax_yz.set_xlabel('Y (px)')
    ax_yz.set_ylabel('Z (px)')
    ax_yz.set_title('Max YZ Projection - Color-coded by X-proximity') # Updated title
    ax_yz.legend()
    ax_yz.set_xlim([min(inlier_y) - 50, max(inlier_y) + 50])
    ax_yz.set_ylim([0, z_size])
    plt.colorbar(scatter_yz, ax=ax_yz, label='X-Proximity') # Colorbar for YZ plot


    # XY Projection (Max)
    ax_xy = axes[2]
    ax_xy.imshow(xy_projection_max, origin='lower', aspect='auto', cmap='gray')
    scatter_xy = ax_xy.scatter(inlier_x, inlier_y, s=5, c=xy_proximity_values_xy, cmap='plasma', label='Inlier Data Points') # Color by XY proximity (Y-coordinate)
    ax_xy.set_xlabel('X (px)')
    ax_xy.set_ylabel('Y (px)')
    ax_xy.set_title('Max XY Projection - Color-coded by Y-proximity') # Updated title
    ax_xy.legend()
    ax_xy.set_xlim([min(inlier_x) - 50, max(inlier_x) + 50])
    ax_xy.set_ylim([min(inlier_y) - 50, max(inlier_y) + 50])
    plt.colorbar(scatter_xy, ax=ax_xy, label='Y-Proximity') # Colorbar for XY plot


    plt.tight_layout()
    plt.show()

def extract_object_data_from_labeled_image(labeled_image_3d):
    """Extracts object data (names, voxel counts, centers, coords) from a 3D labeled image using regionprops."""
    object_data = []
    prop = ['label', 'area', 'centroid', 'coords']
    properties = regionprops_table(labeled_image_3d,properties=prop)
    object_data = pd.DataFrame(properties)
    object_data['Name'] = object_data['label'].apply(lambda x: f"Object_Label_{int(x)}")
    object_data.rename(columns={'label': 'Label', 'area': 'VoxelCount', 'centroid-0': 'Center_Z_px', 'centroid-1': 'Center_Y_px', 'centroid-2': 'Center_X_px', 'coords': 'Coords'}, inplace=True)
    return object_data

def calculate_unocuppied_area(surface_volume_3d, blob_x, blob_y, predicted_z):
    """Calculate the unoccupied area in the surface volume"""
    blob_surface = np.zeros_like(surface_volume_3d)
    blob_surface[blob_y.astype(int), blob_x.astype(int), np.floor(predicted_z).astype(int)] = 1
    total_area = np.sum(surface_volume_3d)
    blob_area = np.sum(blob_surface)
    unoccupied_area = (total_area - blob_area)/total_area
    return (unoccupied_area, blob_surface)
