from utils import fit_polynomial_surface, evaluate_fit, filter_outliers, perform_dbscan_clustering, filter_data_by_cluster, extract_object_data_from_labeled_image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile


    
# --- User Parameters ---
degrees = range(1, 15) # Degrees of polynomial to fit
outlier_threshold = 2.0 # Threshold for outlier detection in std dev
dbscan_eps = 75      # <--- Tunable DBSCAN parameter
dbscan_min_samples = 10 # <--- Tunable DBSCAN parameter
filepath = "example/"
filename = "UV OVA 5d_OT_4-01-objects.ome.tiff"
rmse_values = []
adjusted_r2_values = []

# --- Load Labeled Image and Extract Object Data using regionprops ---
tiff_filepath = filepath + filename 
labeled_image_3d = tifffile.imread(tiff_filepath)
df = extract_object_data_from_labeled_image(labeled_image_3d)
x = df['Center_X_px'].to_numpy().astype(float)
y = df['Center_Y_px'].to_numpy().astype(float)
z = df['Center_Z_px'].to_numpy().astype(float)




# --- DBSCAN Clustering ---
labels, core_samples_mask = perform_dbscan_clustering(x, y, eps=dbscan_eps, min_samples=dbscan_min_samples)

# --- Filter Data by Largest Cluster (Get Indices) ---
cluster_indices = filter_data_by_cluster(labels) # Now returns indices

# --- Use indices to get clustered data ---
clustered_x = x[cluster_indices] # Index original x
clustered_y = y[cluster_indices] # Index original y
clustered_z = z[cluster_indices] # Index original z


    
for degree in degrees:
    model, poly_features, z_predicted, residuals = fit_polynomial_surface(clustered_x, clustered_y, clustered_z, degree)
    # Number of features is the number of coefficients excluding the intercept.
    # PolynomialFeatures includes the intercept term by default, so we subtract 1.
    residuals_std_dev = np.std(residuals)
    residuals_normalized = (residuals-np.mean(residuals)) / residuals_std_dev
    n_features = poly_features.n_output_features_ - 1
    rmse, adjusted_r_squared = evaluate_fit(clustered_z, z_predicted, n_features)
        

    # --- Outlier Filtering ---
    inlier_x, inlier_y, inlier_z, outlier_indices = filter_outliers(clustered_x, clustered_y, clustered_z, residuals_normalized, threshold_std_dev=outlier_threshold)
        
    # --- Refit to Inliers ---
    model_inliers, poly_features_inliers, z_predicted_inliers, residuals_inliers = fit_polynomial_surface(inlier_x, inlier_y, inlier_z, degree)
    print("\n--- Fit to Inliers (After Outlier Removal) ---")
    rmse_inliers, adjusted_r_squared_inliers = evaluate_fit(inlier_z, z_predicted_inliers, poly_features_inliers.n_output_features_ - 1)
    print(f"Degree {degree}: RMSE = {rmse_inliers:.4f}, Adjusted R-squared = {adjusted_r_squared_inliers:.4f}")
    adjusted_r2_values.append(adjusted_r_squared_inliers)
    rmse_values.append(rmse)
      
    
# Plotting RMSE and Adjusted R-squared
plt.figure(figsize=(10, 5))
    
plt.subplot(1, 2, 1) # 1 row, 2 columns, first subplot
plt.plot(degrees, rmse_values, marker='o')
plt.title('RMSE vs Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.xticks(degrees)
plt.grid(True)
    
plt.subplot(1, 2, 2) # 1 row, 2 columns, second subplot
plt.plot(degrees, adjusted_r2_values, marker='o')
plt.title('Adjusted R-squared vs Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Adjusted R-squared')
plt.xticks(degrees)
plt.grid(True)
    
plt.tight_layout() 
plt.show()
