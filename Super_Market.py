import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# Loading DataSet
Data = read_csv('supermarket_sale.csv')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content

# Displaying basic information about the Dataset
print(Data.info())
print(Data.describe())

# Checking missing values
missing_values = Data.isnull().sum()
# Printing Missing values
print("Missing Values:\n", missing_values)

# Converting 'Date' and 'Time' to appropriate data types because date and  time exist
if 'Date' in Data.columns:
    Data['Date'] = pd.to_datetime(Data['Date'])
if 'Time' in Data.columns:
    Data['Time'] = pd.to_datetime(Data['Time']).dt.time

# Identifying nominal attributes (categorical columns)
nominal_columns = Data.select_dtypes(include=['object']).columns
# printing Nominal Attributes with few datas
print("Nominal Attributes:\n", Data[nominal_columns].head(11))

# Converting nominal data to numerical data using Label Encoding
label_encoders = {}
for column in nominal_columns:
    if column != 'Invoice ID':  # Ensuring 'Invoice ID' is not encoded
        le = LabelEncoder()
        Data[column] = le.fit_transform(Data[column])
        label_encoders[column] = le

# Printing all data after conversion to verify
print("Data After Conversion:\n", Data.head(10))

# Separating numerical columns for normalization
numerical_columns = Data.select_dtypes(include=['float64', 'int64']).columns
Data_numerical = Data[numerical_columns]

# Normalizing data (Standardization - Z-score normalization)
scaler = StandardScaler()
Data_normalized = scaler.fit_transform(Data_numerical)

# Converting the normalized data back to DataFrame
Data_normalized = pd.DataFrame(Data_normalized, columns=numerical_columns)

# Combining normalized numerical data with non-numerical data
Data_combined = Data.copy()
Data_combined[numerical_columns] = Data_normalized

# Verifying the normalized data
print("Normalized Data:\n", Data_combined.tail(11))

# Function to perform K-means clustering and plot results
def perform_kmeans(data, k, distance_metric):
    if distance_metric == 'euclidean':
        kmeans = KMeans(n_clusters=k, random_state=42, algorithm='lloyd')
        clusters = kmeans.fit_predict(data)
    elif distance_metric == 'manhattan':
        kmeans = KMeans(n_clusters=k, random_state=42, algorithm='lloyd')
        data = pairwise_distances(data, metric='manhattan')
        clusters = kmeans.fit_predict(data)
    else:
        raise ValueError("Unsupported distance metric")

    data_df = pd.DataFrame(data)
    data_df[f'Cluster_{k}_{distance_metric}'] = clusters
    output_file = f'kmeans_k{k}_{distance_metric}.csv'
    data_df.to_csv(output_file, index=False)
    print(f"K-means clustering with K={k} and {distance_metric} distance saved to {output_file}")

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=data_df, x=data_df.columns[0], y=data_df.columns[1], hue=f'Cluster_{k}_{distance_metric}', palette='viridis')
    plt.title(f'K-Means Clustering with K={k} and {distance_metric} Distance')
    plt.savefig(f'kmeans_k{k}_{distance_metric}.png')
    plt.show()

# Function to determine the optimal number of clusters using the elbow method
def plot_elbow_method(data, distance_metric):
    distortions = []
    K = range(1, 10)
    for k in K:
        if distance_metric == 'euclidean':
            kmeans = KMeans(n_clusters=k, random_state=42, algorithm='lloyd')
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        elif distance_metric == 'manhattan':
            kmeans = KMeans(n_clusters=k, random_state=42, algorithm='lloyd')
            data_dist = pairwise_distances(data, metric='manhattan')
            kmeans.fit(data_dist)
            distortions.append(kmeans.inertia_)
        else:
            raise ValueError("Unsupported distance metric")

    plt.figure(figsize=(10, 7))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(f'Elbow Method showing the optimal k ({distance_metric} distance)')
    plt.savefig(f'elbow_method_{distance_metric}.png')
    plt.show()

# Testing with different K values and distance metrics
k_values = [2,3, 4, 5]
distance_metrics = ['euclidean', 'manhattan']

for distance_metric in distance_metrics:
    plot_elbow_method(Data_normalized, distance_metric)
    for k in k_values:
        perform_kmeans(Data_normalized, k, distance_metric)
