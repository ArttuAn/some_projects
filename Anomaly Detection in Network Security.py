from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("log2.csv")

print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values for each feature:\n", missing_values)

# To check if there are any missing values at all
if missing_values.sum() > 0:
    print("There are missing values in the dataset.")
else:
    print("No missing values in the dataset.")

# Renaming the columns
df.columns = ['source port', 'destination port', 'nat source port', 'nat destination port', 'action',
              'bytes', 'bytes sent', 'bytes received', 'packets', 'elapsed time (s)', 'pkts sent', 'pkts received']

# Check inconsistencies
def check_inconsistencies(column_to_check, col1, col2, df):
    inconsistent_rows = df[df[column_to_check] != (df[col1] + df[col2])]

    if not inconsistent_rows.empty:
        print(
            f"Rows where '{column_to_check}' is not the sum of '{col1}' and '{col2}':")
        print(inconsistent_rows)
    else:
        print(
            f"'{column_to_check}' is always the sum of '{col1}' and '{col2}' in every row.")


check_inconsistencies('bytes', 'bytes sent', 'bytes received', df)
check_inconsistencies('packets', 'pkts sent', 'pkts received', df)

# Pie Chart for 'Action'
plt.figure(figsize=(6, 6))
action_counts = df['action'].value_counts()
plt.pie(action_counts, labels=action_counts.index,
        autopct='%1.1f%%', startangle=90)
plt.title('Actions Distribution')
plt.show()


# Drop the columns bytes and packets since they dont provide any additional value, they're always the sums of bytes+bytes sent and pkts sent+pkts received
df.drop(columns=['bytes', 'packets'],
        inplace=True)

# Normalization for continuous values
cols_to_normalize = ['source port', 'destination port', 'nat source port', 'nat destination port',
                     'bytes sent', 'bytes received', 'elapsed time (s)', 'pkts sent', 'pkts received']

scaler = MinMaxScaler()

# Fit and transform the data
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# One-hot encoding for 'Action'
df = pd.get_dummies(df, columns=['action'], prefix=['action'])

# Heatmap for Correlation Matrix
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

################################################################################################################################################################
############################################################# K-MEANS ##########################################################################################
################################################################################################################################################################

# To determine the best number of clusters, we can use the Elbow method
inertia_list = []
for i in range(1, 11):  # I'll try for 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans.fit(df)
    inertia_list.append(kmeans.inertia_)
    print(f"For {i} clusters, inertia: {kmeans.inertia_}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia_list, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal Cluster Number')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Train the KMeans model with 3 clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df)

# Visualizing the clusters
# Using PCA for 2D visualization due to multiple featues. This provides an approximation of our multi-dimensional data in two dimensions.
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
principalDf['cluster'] = df['cluster']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=principalDf, x='PC1', y='PC2',
                hue='cluster', palette='viridis', s=100, alpha=0.7)
plt.title('K-means PCA Cluster Visualization')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.show()


# Interpreting the results
cluster_summary = df.groupby('cluster').mean()
print(cluster_summary)

kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_3.fit(df)
print(f"Inertia for 3 clusters: {kmeans_3.inertia_}")


# Define a function to predict the cluster for new data
def predict_cluster(new_data, kmeans_model, scaler):

    # Check if 'action' exists in new_data and if it does, one-hot encode it
    if 'action' in new_data.columns:
        new_data = pd.get_dummies(
            new_data, columns=['action'], prefix=['action'])

    # Make sure new_data has all the columns in the original df (excluding 'cluster')
    for col in df.columns:
        if col not in new_data.columns and col != 'cluster':
            new_data[col] = 0

    # Order columns to match original df
    new_data = new_data[df.columns.drop('cluster')]

    # Scale the new data
    new_data[cols_to_normalize] = scaler.transform(new_data[cols_to_normalize])

    # Predict cluster label
    cluster_label = kmeans_model.predict(new_data)

    return cluster_label


# Test the function with a sample data point
sample_data = pd.DataFrame({
    'source port': [57222],
    'destination port': [53],
    'nat source port': [54587],
    'nat destination port': [53],
    'action': ['allow'],
    'bytes sent': [94],
    'bytes received': [83],
    'elapsed time (s)': [30],
    'pkts sent': [1],
    'pkts received': [1]
})

predicted_cluster = predict_cluster(sample_data, kmeans, scaler)
print(f"Predicted cluster for the sample data: {predicted_cluster[0]}")


# After you've computed your clusters using KMeans
silhouette_avg = silhouette_score(df, df['cluster'])

print(f"K-Means Silhouette Score for {optimal_clusters} clusters: {silhouette_avg}")


# Compute Davies-Bouldin Index for KMeans
db_score_kmeans = davies_bouldin_score(
    df.drop(['cluster'], axis=1), df['cluster'])
print(f'Davies-Bouldin Index for KMeans: {db_score_kmeans}')

################################################################################################################################################################
############################################################# DBSCAN ##########################################################################################
################################################################################################################################################################

print("for jupyter:")

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(df)


principalDf['dbscan_cluster'] = df['dbscan_cluster']

distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                   '#d62728', '#9467bd', '#8c564b', '#e377c2']


plt.figure(figsize=(10, 6))
sns.scatterplot(data=principalDf, x='PC1', y='PC2',
                hue='dbscan_cluster', palette=distinct_colors, s=100, alpha=0.7)
plt.title('DBSCAN PCA Cluster Visualization')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.show()


dbscan_cluster_summary = df.groupby('dbscan_cluster').mean()
print(dbscan_cluster_summary)


def predict_dbscan_cluster(new_data, dbscan_model, scaler):
    # Check if 'action' exists in new_data and if it does, one-hot encode it
    if 'action' in new_data.columns:
        new_data = pd.get_dummies(
            new_data, columns=['action'], prefix=['action'])

    # Make sure new_data has all the columns in the original df (excluding 'dbscan_cluster')
    for col in df.columns:
        if col not in new_data.columns and col != 'dbscan_cluster':
            new_data[col] = 0

    # Order columns to match original df
    new_data = new_data[df.columns.drop('dbscan_cluster')]

    # Scale the new data
    new_data[cols_to_normalize] = scaler.transform(new_data[cols_to_normalize])

    # Predict cluster label (-1 for noise) using DBSCAN
    cluster_label = dbscan_model.fit_predict(new_data)
    return cluster_label


# Predict DBSCAN cluster label for the sample data
predicted_dbscan_cluster = predict_dbscan_cluster(sample_data, dbscan, scaler)

# Print the predicted cluster label
print(
    f"Predicted DBSCAN cluster for the sample data: {predicted_dbscan_cluster[0]}")

# Exclude noise points (cluster label -1) from evaluation as many metrics are undefined for single-sample clusters
filtered_data = df[df['dbscan_cluster'] != -1]

# Compute Silhouette Score
sil_score = silhouette_score(filtered_data.drop(
    'dbscan_cluster', axis=1), filtered_data['dbscan_cluster'])
print(f'Silhouette Score: {sil_score}')

# Compute Davies-Bouldin Index
db_score = davies_bouldin_score(filtered_data.drop(
    'dbscan_cluster', axis=1), filtered_data['dbscan_cluster'])
print(f'Davies-Bouldin Index: {db_score}')
