import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


class LiftAnomalyModel:
    def __init__(self):
        self.all_trips = None
        self.combined_labels = None

    def normalize_time(self, df, target_length=2000):
        original_time = np.linspace(0, len(df) - 1, len(df))
        normalized_time = np.linspace(0, len(df) - 1, target_length)
        return normalized_time

    def extract_trip_features(self, df, target_length=2000):
        normalized_time = self.normalize_time(df, target_length)
        z_axis_resampled = np.interp(normalized_time, np.linspace(0, len(df) - 1, len(df)), df['Acceleration (m/s²)'])
        dt = np.mean(np.diff(normalized_time))
        velocity = np.gradient(z_axis_resampled, dt)
        acceleration = np.gradient(velocity, dt)
        jerk = np.gradient(acceleration, dt)
        return pd.DataFrame({'timestamp': np.linspace(0, 20, target_length), 'z_axis': z_axis_resampled}), velocity, acceleration, jerk

    def extract_shape_features(self, trip):
        z_axis = trip[1]
        total_displacement = np.sum(np.abs(z_axis))  # Total area under the curve
        mean_value = np.mean(z_axis)
        variance_value = np.var(z_axis)  # indicating stability or erratic behavior
        skewness = skew(z_axis)  # indicating if the data is tilted towards higher or lower values
        kurt = kurtosis(z_axis)  # indicating frequent extreme values
        peak_to_peak = np.ptp(z_axis)  # Peak-to-Peak distance, range of acceleration
        return [total_displacement, mean_value, variance_value, skewness, kurt, peak_to_peak]

    def predict(self, file_paths):
        data_list = []
        for file_path in file_paths:
            data = pd.read_csv(file_path)
            data.columns = ['Unnamed: 0', 'Acceleration (m/s²)']
            trip_features = self.extract_trip_features(data, target_length=2000)
            data_list.append((trip_features, file_path))
    
        # Shuffle the data
        np.random.shuffle(data_list)

        self.all_trips = [item[0] for item in data_list]
        file_names = [item[1] for item in data_list]

        for i in range(len(self.all_trips)):
            self.all_trips[i] = (self.all_trips[i][0], self.all_trips[i][1] - np.mean(self.all_trips[i][1]), self.all_trips[i][2], self.all_trips[i][3])

        features = np.array([self.extract_shape_features(trip) for trip in self.all_trips])

        if features.size == 0:
            return {"error": "No data to process. Ensure the files contain valid CSV data."}

        optimal_clusters = 3
        kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)

        iso_forest = IsolationForest(contamination=0.3, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features)

        self.combined_labels = np.where(anomaly_labels == -1, -1, labels)

        clean_results = []
        anomaly_results = []
        
        for i, (trip, label) in enumerate(zip(self.all_trips, self.combined_labels)):
            # Ensure the 'cluster' column is added before accessing it
            trip[0]['cluster'] = self.combined_labels[i]
            if self.combined_labels[i] == -1:
                anomaly_results.append({"file": file_names[i], "cluster": int(trip[0]['cluster'].iloc[0]), "anomaly": True})
            else:
                clean_results.append({"file": file_names[i], "cluster": int(trip[0]['cluster'].iloc[0]), "anomaly": False})
        
        return anomaly_results


    def generate_plot(self):
        plt.figure(figsize=(15, 5))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
        cluster_names = {0: "Clean", 1: "Clean", -1: "Anomaly", 2: "Clean"}
        unique_labels = set(self.combined_labels)
        legend_handles = []

        for label in unique_labels:
            for i, trip in enumerate(self.all_trips):
                if self.combined_labels[i] == label:
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
        plot_path = 'temp_files/clustering_plot.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Print the path of the plot
        print(f"Plot saved at: {plot_path}")