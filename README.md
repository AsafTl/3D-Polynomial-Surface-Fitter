# 3D Polynomial Surface Fitter for Object Centroids
Used in Ludin et al. [CRATER tumor niches facilitate CD8+ T cell engagement and correspond with immunotherapy success] Cell 2025 to calculate attrition of tumor surface in OT model
This project provides a suite of Python scripts to fit a 3D polynomial surface to a point cloud derived from object centroids in a 3D labeled image. It includes functionality for data clustering, outlier removal, model evaluation, and visualization.

The primary goal is to generate a smooth mathematical representation of a biological surface (like a tumor boundary) from microscopy data.

If you use this package, please cite:
Ludin et al. Cell 2025

## Features

-   **Data Extraction**: Loads 3D labeled `.ome.tiff` images and extracts object centroids using `skimage.measure.regionprops_table`.
-   **Clustering**: Uses **DBSCAN** to isolate the main cluster of points, separating the primary surface from noise.
-   **Surface Fitting**: Fits a polynomial surface of a user-defined degree to the clustered 3D points.
-   **Outlier Removal**: Identifies and removes outliers based on the standard deviation of residuals from the initial fit.
-   **Evaluation**: Calculates Root Mean Squared Error (RMSE) and Adjusted R-squared to evaluate the goodness of fit across different polynomial degrees.
-   **Visualization**: Generates 2D projections (XY, XZ, YZ) to visually inspect the fitted surface against the data points.
-   **Output**: Saves the final fitted surface as a 3D TIFF volume and provides model parameters.

## File Structure

```
├── main/
│   ├── main.py
│   ├── Eval_fit.py
│   ├── utils.py
│   └── example/
│        └── UV OVA 5d_OT_4-01-objects.ome.tiff  (Example Input)

├── requirements.txt
└── README.md
```

## Installation

1.  **Clone the repository:**
2.  **Install the required Python packages:**
## Usage

The project contains two main executable scripts: `Eval_fit.py` for finding the optimal polynomial degree and `main.py` for performing the final fit and generating the output surface.

### 1. Evaluate the Best Polynomial Degree

First, run `Eval_fit.py` to determine the best polynomial degree for your data. This script iterates through a range of degrees, fits a model for each, and plots the **RMSE** and **Adjusted R-squared** values.

-   **Configure `Eval_fit.py`**:
    -   Set the `filepath` and `filename` variables to point to your input `.ome.tiff` file.
    -   Adjust the `degrees` range and `outlier_threshold` as needed, the smaller 'outlier_threshold' the more agressive filtering
-   **Run the script:**
-   **Analyze the plots**: Look for the "elbow point" in the RMSE plot or the point where the Adjusted R-squared value plateaus or max. This indicates the degree where the model stops improving significantly.

### 2. Generate the Final Surface

Once you have chosen a polynomial degree, use `main.py` to generate the final surface.

-   **Configure `main.py`**:
    -   Set `degree_to_use` to your chosen degree.
    -   Verify `filepath`, `filename`, and `outlier_threshold` are correct.
    -   Set an `output_prefix` for the output files.
-   **Run the script:**
    

### Main Parameters

You can tune the following parameters in both `main.py` and `Eval_fit.py`:

-   `degree_to_use` / `degrees`: The polynomial degree(s) for the surface fit.
-   `outlier_threshold`: The number of standard deviations from the mean residual to classify a point as an outlier. A value of `2.0` is common.
-   `dbscan_eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is a key DBSCAN parameter.
-   `dbscan_min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.

## Output

The `main.py` script will generate the following files in your data directory:

1.  **Fitted Surface Volume (`..._surface_degree_6.tif`)**: A 3D TIFF file where voxels on the fitted polynomial surface have a value of 1.
2.  **Plots**: Matplotlib windows will display residual histograms and 2D projections of the fitted surface for visual inspection.