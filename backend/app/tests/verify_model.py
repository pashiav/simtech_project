import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import zipfile

def unzip_files(zip_file_path, extraction_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
        print(f"Extracted all files to {extraction_path}")

zip_file_path = 'data/test_files.zip'
extraction_path = 'data/test_files'
unzip_files(zip_file_path, extraction_path)

all_files = [f for f in os.listdir(extraction_path) if os.path.isfile(os.path.join(extraction_path, f))]

# Function to normalize the time axis
def normalize_time(df, target_length=2000):
    original_time = np.linspace(0, len(df) - 1, len(df))
    normalized_time = np.linspace(0, len(df) - 1, target_length)
    return normalized_time

# Function to extract features for each trip
def extract_trip_features(df, target_length=2000):
    normalized_time = normalize_time(df, target_length)
    z_axis_resampled = np.interp(normalized_time, np.linspace(0, len(df) - 1, len(df)), df['Acceleration (m/s²)'])
    dt = np.mean(np.diff(normalized_time))
    velocity = np.gradient(z_axis_resampled, dt)
    acceleration = np.gradient(velocity, dt)
    jerk = np.gradient(acceleration, dt)
    return pd.DataFrame({'timestamp': np.linspace(0, 20, target_length), 'z_axis': z_axis_resampled}), velocity, acceleration, jerk

# Load and normalize data
data_list = []

for file in all_files:
    data = pd.read_csv(os.path.join(extraction_path, file))
    data.columns = ['Unnamed: 0', 'Acceleration (m/s²)']
    trip_features = extract_trip_features(data, target_length=2000)
    data_list.append((trip_features, file))

# Shuffle the data
np.random.shuffle(data_list)

# Extract the trips and file names
all_trips = [item[0] for item in data_list]
file_names = [item[1] for item in data_list]

# Center each trip around zero
for i in range(len(all_trips)):
    all_trips[i] = (all_trips[i][0], all_trips[i][1] - np.mean(all_trips[i][1]), all_trips[i][2], all_trips[i][3])

# Extract shape-based features for clustering
def extract_shape_features(trip):
    z_axis = trip[1]
    total_displacement = np.sum(np.abs(z_axis))  # Total area under the curve
    mean_value = np.mean(z_axis)
    variance_value = np.var(z_axis) # indicating stability or erratic behavior
    skewness = skew(z_axis) # indicating if the data is tilted towards higher or lower values
    kurt = kurtosis(z_axis) # indicating frequent extreme values
    peak_to_peak = np.ptp(z_axis)  # Peak-to-Peak distance, range of acceleration
    return [total_displacement, mean_value, variance_value, skewness, kurt, peak_to_peak]

features = np.array([extract_shape_features(trip) for trip in all_trips])

# Check if features array is empty
if features.size == 0:
    print("No data to process. Ensure the zip file contains valid CSV files.")
    exit()

# Apply K-Means clustering on the shape features with 3 clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
labels = kmeans.fit_predict(features)

# Apply Isolation Forest for anomaly detection with increased contamination rate
iso_forest = IsolationForest(contamination=0.3, random_state=42)  # Increased contamination rate to 0.3
anomaly_labels = iso_forest.fit_predict(features)

# Combine K-Means labels with anomaly detection
combined_labels = np.where(anomaly_labels == -1, -1, labels)

# Add combined cluster labels to each trip
for i, trip in enumerate(all_trips):
    trip[0]['cluster'] = combined_labels[i]

# Plot each trip with different colors for each cluster
plt.figure(figsize=(15, 5))

# Ensure distinct visualization for each cluster without changing colors
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
cluster_names = {0: "Cluster 0", 1: "Cluster 1", -1: "Anomaly", 2: "Cluster 2"}
unique_labels = set(combined_labels)
legend_handles = []

for label in unique_labels:
    for i, trip in enumerate(all_trips):
        if combined_labels[i] == label:
            color = 'k--' if label == -1 else colors[label % len(colors)]
            name = cluster_names.get(label, f'Cluster {label}')
            if name not in [h.get_label() for h in legend_handles]:
                line, = plt.plot(trip[0]['timestamp'], trip[0]['z_axis'], color, label=name)
                legend_handles.append(line)
            else:
                plt.plot(trip[0]['timestamp'], trip[0]['z_axis'], color)

plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Shape-Based Clustering of Elevator Trips with Anomaly Detection')
plt.legend(handles=legend_handles)
plt.grid(True)
plt.show()

# Display cluster labels
print("Cluster labels for each trip:", combined_labels)

# Identify and print trips marked as anomalies
for i, (trip, label) in enumerate(zip(all_trips, combined_labels)):
    if combined_labels[i] == -1:
        print(f"Trip {i} (File: {file_names[i]}) is marked as an anomaly with cluster label {trip[0]['cluster'].iloc[0]}")
