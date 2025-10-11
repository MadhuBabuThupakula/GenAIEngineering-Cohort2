# Import necessary libraries
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

# --- 1. Generate Sample High-Dimensional Data ---
# Let's create some synthetic 768-dimensional data for demonstration.
# In a real scenario, you would replace this with your actual dataset.

# Number of data points
n_samples = 200
# Original dimensionality
n_features = 768
# Number of distinct clusters/groups in the data (for visualization purposes)
n_clusters = 4

# Generate random data, attempting to create some loose clusters
# For simplicity, we'll generate data from different multivariate normal distributions
# centered at different points.
data_768d = []
labels = [] # To color points by their original cluster

np.random.seed(42) # for reproducibility

for i in range(n_clusters):
    # Create a random center for each cluster
    center = np.random.rand(n_features) * 10 * (i + 1)
    # Generate points around this center
    cluster_data = np.random.multivariate_normal(
        mean=center,
        cov=np.eye(n_features) * 5, # Add some variance
        size=n_samples // n_clusters
    )
    data_768d.extend(cluster_data)
    labels.extend([i] * (n_samples // n_clusters))

data_768d = np.array(data_768d)
labels = np.array(labels)

print(f"Shape of original data: {data_768d.shape}")
print(f"Shape of labels: {labels.shape}")

# --- 2. Apply t-SNE for Dimensionality Reduction ---
# We want to reduce the data from 768 dimensions to 3 dimensions.

print("\nApplying t-SNE... (This might take a moment for high-dimensional data)")

# Initialize t-SNE
# - n_components: The dimension of the embedded space (3 for 3D).
# - perplexity: Related to the number of nearest neighbors. Typical values are between 5 and 50.
# - n_iter: Number of iterations for optimization.
# - learning_rate: Usually between 10 and 1000.
# - random_state: For reproducibility.
tsne = TSNE(n_components=3,
            perplexity=30.0, # A common default, adjust based on your dataset size
            n_iter=1000,     # Increase if the embedding doesn't converge well
            learning_rate='auto', # 'auto' is available in newer scikit-learn versions (>=1.0.0)
                                 # otherwise, try 200
            init='pca',      # PCA initialization can be more stable
            random_state=42)

# Fit and transform the data
data_3d = tsne.fit_transform(data_768d)

print(f"Shape of transformed data (3D): {data_3d.shape}")

# --- 3. Plot the 3D Transformed Data using Matplotlib ---
print("\nPlotting the 3D data...")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
# We use the 'labels' generated earlier to color-code the points if available.
# If you don't have predefined labels for your actual data, you can plot all points in one color.
scatter = ax.scatter(data_3d[:, 0],  # x-coordinates
                     data_3d[:, 1],  # y-coordinates
                     data_3d[:, 2],  # z-coordinates
                     c=labels,       # Color by original cluster label
                     cmap=plt.cm.get_cmap("viridis", n_clusters)) # Colormap

# Set labels and title
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_zlabel("t-SNE Component 3")
ax.set_title("3D t-SNE projection of 768-dimensional data")

# Add a color bar (legend for clusters)
# This is more meaningful if you have actual class labels
if n_clusters > 1:
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 label=f'Cluster {i+1}',
                                 markerfacecolor=plt.cm.get_cmap("viridis", n_clusters)(i / (n_clusters -1 if n_clusters > 1 else 1) ),
                                 markersize=8) for i in range(n_clusters)]
    ax.legend(handles=legend_elements, title="Clusters")
else:
    # If no specific clusters, just show a generic point
    ax.scatter([], [], [], c='black', label='Data Point') # Dummy for legend
    ax.legend()


# Improve layout and show plot
plt.tight_layout()
plt.show()

print("\nDone. The plot should display the 3D representation of your data.")
print("Note: t-SNE is a visualization technique; the distances between clusters in the low-dimensional space")
print("are not always perfectly representative of distances in the high-dimensional space.")
print("Experiment with perplexity and other parameters for best results on your specific dataset.")
