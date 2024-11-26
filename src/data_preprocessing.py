import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering


# data_preprocessing.py
class Imputer:
    def __init__(self, similarity_threshold=0.6, majority_threshold=0.5):
        self.similarity_threshold = similarity_threshold
        self.majority_threshold = majority_threshold

    def compute_similarity(self, features1, features2):
        normalized_f1 = tf.nn.l2_normalize(features1, axis=1)
        normalized_f2 = tf.nn.l2_normalize(features2, axis=1)
        return tf.matmul(normalized_f1, normalized_f2, transpose_b=True)

    def impute_attributes(self, df, n_attributes):
        """
        Impute missing attributes based on visual similarity clustering.
        """
        features = np.array(list(df['image_features_resnet'].values))
        similarities = self.compute_similarity(
            tf.constant(features), tf.constant(features)
        ).numpy()
        distances = 1 - similarities
        np.fill_diagonal(distances, 0)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.similarity_threshold,
            affinity="precomputed",
            linkage="complete"
        )
        clusters = clustering.fit_predict(distances)
        cluster_groups = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(idx)

        for cluster_id, indices in cluster_groups.items():
            if len(indices) > 1:
                cluster_data = df.iloc[indices]
                for attr in [f"attr_{i}" for i in range(1, n_attributes + 1)]:
                    valid_values = cluster_data[attr].dropna()
                    if valid_values.size > 0:
                        most_common_value = valid_values.mode()[0]
                        df.loc[indices, attr] = df.loc[indices, attr].fillna(most_common_value)
        return df