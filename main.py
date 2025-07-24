from utils import fit_polynomial_surface, evaluate_fit, plot_residuals_histogram, filter_outliers, perform_dbscan_clustering, filter_data_by_cluster, create_surface_volume_optimized, visualize_surface_fit, extract_object_data_from_labeled_image, predict_blob_surface, calculate_unocuppied_area  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile

# --- User Parameters ---
degree_to_use = 6 # Set degree
outlier_threshold = 2.0 # Threshold for outlier detection in std dev
dbscan_eps = 75      # <--- Tunable DBSCAN parameter
dbscan_min_samples = 10 # <--- Tunable DBSCAN parameter
filepath = "example/" # Path to the directory containing the TIFF file
filename = "UV OVA 5d_OT_4-01-objects.ome.tiff"
output_prefix = f"tumor_surface_degree_{degree_to_use}" # Prefix for output files



tiff_filepath = filepath + filename 

# --- Load Labeled Image and Extract Object Data using regionprops ---
labeled_image_3d = tifffile.imread(tiff_filepath)
df = extract_object_data_from_labeled_image(labeled_image_3d)
x = df['Center_X_px'].to_numpy().astype(float)
y = df['Center_Y_px'].to_numpy().astype(float)
z = df['Center_Z_px'].to_numpy().astype(float)
z_size = labeled_image_3d.shape[0] # Get z size from image


if x is not None: # Proceed only if data loading was successful
    # --- DBSCAN Clustering ---
    labels, core_samples_mask = perform_dbscan_clustering(x, y, eps=dbscan_eps, min_samples=dbscan_min_samples)

    # --- Filter Data by Largest Cluster (Get Indices) ---
    cluster_indices = filter_data_by_cluster(labels) # Returns indices

    if cluster_indices is not None: # Proceed if clustering found a cluster
        # --- Use indices to get clustered data ---
        clustered_x = x[cluster_indices] # Index original x
        clustered_y = y[cluster_indices] # Index original y
        clustered_z = z[cluster_indices] # Index original z
        clustered_coords = df.loc[cluster_indices,'Coords'].reset_index(drop=True) # Index original coords
        

    

    # --- Initial Fit and Residual Analysis ---
    model, poly_features, z_predicted, residuals = fit_polynomial_surface(clustered_x, clustered_y, clustered_z, degree_to_use)
    residuals_std_dev = np.std(residuals)
    residuals_normalized = (residuals-np.mean(residuals)) / residuals_std_dev

    print("--- Initial Fit (Before Outlier Removal) ---")
    rmse_initial, adjusted_r_squared_initial = evaluate_fit(clustered_z, z_predicted, poly_features.n_output_features_ - 1)
    print(f"Degree {degree_to_use}: RMSE = {rmse_initial:.4f}, Adjusted R-squared = {adjusted_r_squared_initial:.4f}")
    plot_residuals_histogram(residuals_normalized, degree_to_use, dataset_label="Before Outlier Removal") # Label for clarity
    # --- Outlier Filtering ---
    inlier_x, inlier_y, inlier_z, inlier_indices = filter_outliers(clustered_x, clustered_y, clustered_z, residuals_normalized, threshold_std_dev=outlier_threshold)
    
    used_coords = clustered_coords[inlier_indices].reset_index(drop=True)
    coords = np.concatenate(used_coords.tolist(),axis=0)
    
    
    # --- Refit to Inliers ---
    model_inliers, poly_features_inliers, z_predicted_inliers, residuals_inliers = fit_polynomial_surface(inlier_x, inlier_y, inlier_z, degree_to_use)
    residuals_inliers_std_dev = np.std(residuals_inliers)
    residuals_inliers_normalized = (residuals_inliers-np.mean(residuals_inliers)) / residuals_inliers_std_dev
    # --- Evaluate and Plot Refitted Surface ---
    print("\n--- Fit to Inliers (After Outlier Removal) ---")
    rmse_inliers, adjusted_r_squared_inliers = evaluate_fit(inlier_z, z_predicted_inliers, poly_features_inliers.n_output_features_ - 1)
    print(f"Degree {degree_to_use}: RMSE = {rmse_inliers:.4f}, Adjusted R-squared = {adjusted_r_squared_inliers:.4f}")
    plot_residuals_histogram(residuals_inliers_normalized, degree_to_use, dataset_label="After Outlier Removal") # Label for clarity

    print("\n--- Outlier Indices (removed from original data) ---")
   
    # --- Create Surface Volume ---
    
    surface_volume_3d = create_surface_volume_optimized(inlier_x, inlier_y, model_inliers, poly_features_inliers, z_size)

    print("\n--- Surface Volume Created ---")
    print("Surface volume shape:", surface_volume_3d.shape)
    print("Surface volume data type:", surface_volume_3d.dtype)

    # --- Visualize Surface Fit ---
    visualize_surface_fit(surface_volume_3d, inlier_x, inlier_y, inlier_z) # Call visualization function

    z_blob_predict = predict_blob_surface(coords[:,2], coords[:,1], poly_features_inliers, model_inliers)
    unoccupied_area, blob_surface = calculate_unocuppied_area(surface_volume_3d, coords[:,2], coords[:,1], z_blob_predict)

    print(f"Unoccupied area: {unoccupied_area:.2f}")
    

    # --- Output Surface Parameters ---
    #print("\n--- Surface Model Parameters (Degree 6) ---")
    coefficients = model_inliers.coef_
    intercept = model_inliers.intercept_
    #print("Coefficients:", coefficients)
    #print("Intercept:", intercept)

    # --- Save Surface Volume to TIFF ---
    tiff_filename = filepath+filename.split('.')[0]+'_'+output_prefix+".tif"
    surface_volume_3d_transposed = np.transpose(surface_volume_3d, (2, 0, 1)) # Transpose to X, Y, Z order
    tifffile.imwrite(tiff_filename, surface_volume_3d_transposed)
    print(f"\nSurface volume saved to: {tiff_filename} with shape {surface_volume_3d_transposed.shape}") # Print shape after transpose
else:
    print("Data loading failed. Script execution stopped.")