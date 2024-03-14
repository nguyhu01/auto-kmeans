import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def handle_missing_values(data):
    data.dropna(inplace=True)
    return data

def scale_data(data, data_is_normal):
    """
    data: a DataFrame
    """
    print(type(data))
    num_col = data.select_dtypes(exclude=['object']).columns

    if data_is_normal:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(data[num_col])
    scaled_data_df = pd.DataFrame(scaled_data, columns=num_col)
    return scaled_data, scaled_data_df

def visualize_heatmap(data):
    plt.subplots(figsize=(20,15))
    sns.heatmap(data.corr(), annot=True, cmap='PuBu')
    plt.show()

def visualize_histograms(data):
    data.hist(bins=15, figsize=(20, 15), layout=(5, 4))
    plt.gca().spines[['top', 'right',]].set_visible(False)
    plt.show()

def visualize_3d_clusters(data, cluster_labels, f1, f2, f3):
    PLOT = go.Figure()
    for i in list(cluster_labels.unique()):
        PLOT.add_trace(go.Scatter3d(x=data[cluster_labels== i][f1],
                                    y=data[cluster_labels== i][f2],
                                    z=data[cluster_labels== i][f3],
                                    mode='markers', marker_size=6, marker_line_width=1,
                                    name=str(i)))
    PLOT.update_traces(hovertemplate='{f1}: %{x} <br>{f2}: %{y} <br>{f3}: %{z}')
    PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
                      scene=dict(xaxis=dict(title=str(f1), titlefont_color='black'),
                                 yaxis=dict(title=str(f2), titlefont_color='black'),
                                 zaxis=dict(title=str(f3), titlefont_color='black')),
                      font=dict(family="Gilroy", color='black', size=12))
    PLOT.show()

def visualize_cluster_centers(cluster_centers):
    plt.figure(figsize=(10, 6))
    for i in range(len(cluster_centers)):
        plt.plot(cluster_centers[i], label=f'Cluster {i+1}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.title('Cluster Centers')
    plt.legend()
    plt.show()

def plot_feature_distribution_by_cluster(data, cluster_labels, feature_name):
    plt.figure(figsize=(10, 6))
    for cluster_label in range(len(cluster_centers)):
        cluster_data = data[cluster_labels == cluster_label]
        sns.kdeplot(cluster_data[feature_name], label=f'Cluster {cluster_label+1}')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.title(f'Distribution of {feature_name} by Cluster')
    plt.legend()
    plt.show()

def plot_feature_boxplot_by_cluster(data, cluster_labels, feature_name):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Clusters', y=feature_name, data=data)
    plt.xlabel('Cluster')
    plt.ylabel(feature_name)
    plt.title(f'Boxplot of {feature_name} by Cluster')
    plt.show()

def plot_feature_scatter_by_cluster(data, cluster_labels, feature_name1, feature_name2):
    plt.figure(figsize=(10, 6))
    for cluster_label in range(len(cluster_centers)):
        cluster_data = data[cluster_labels == cluster_label]
        plt.scatter(cluster_data[feature_name1], cluster_data[feature_name2], label=f'Cluster {cluster_label+1}')
    plt.xlabel(feature_name1)
    plt.ylabel(feature_name2)
    plt.title(f'Scatter Plot of {feature_name1} vs {feature_name2} by Cluster')
    plt.legend()
    plt.show()

def plot_silhouette_scores(data, cluster_labels, k_range):
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(data, labels))

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xticks(k_range)
    plt.show()

def plot_elbow_method(data, k_range):
    inertia_values = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_values, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method - Inertia vs Number of Clusters')
    plt.xticks(k_range)
    plt.show()

