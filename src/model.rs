//! K-Means clustering model implementation

use crate::data::RfmData;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_nn::distance::L2Dist;
use ndarray::{Array1, Array2};

/// K-Means model wrapper with fitted parameters
#[derive(Debug)]
pub struct KMeansModel {
    /// Fitted K-Means model from linfa
    pub model: KMeans<f64, L2Dist>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Cluster assignments for training data
    pub labels: Array1<usize>,
    /// Cluster centroids in normalized space
    pub centroids: Array2<f64>,
    /// Within-cluster sum of squares (inertia)
    pub inertia: f64,
}

impl KMeansModel {
    /// Predict cluster for new data point
    pub fn predict(&self, features: &Array1<f64>) -> crate::Result<usize> {
        if features.len() != 3 {
            anyhow::bail!("Feature vector must have exactly 3 dimensions");
        }

        // Find nearest centroid
        let mut min_distance = f64::INFINITY;
        let mut closest_cluster = 0;

        for (cluster_idx, centroid) in self.centroids.outer_iter().enumerate() {
            let distance: f64 = features
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if distance < min_distance {
                min_distance = distance;
                closest_cluster = cluster_idx;
            }
        }

        Ok(closest_cluster)
    }

    /// Get cluster sizes
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.n_clusters];
        for &label in self.labels.iter() {
            if label < self.n_clusters {
                sizes[label] += 1;
            }
        }
        sizes
    }

    /// Compute basic silhouette coefficient for a subset of points (for efficiency)
    pub fn compute_silhouette_sample(&self, features: &Array2<f64>, sample_size: usize) -> f64 {
        let n_samples = features.nrows().min(sample_size);
        if n_samples < 2 {
            return 0.0;
        }

        let mut silhouette_sum = 0.0;

        for i in 0..n_samples {
            let point = features.row(i);
            let cluster_label = self.labels[i];

            // Calculate a(i): mean distance to points in same cluster
            let mut same_cluster_distances = Vec::new();
            let mut other_cluster_distances: Vec<Vec<f64>> = vec![Vec::new(); self.n_clusters];

            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let other_point = features.row(j);
                let distance = euclidean_distance(&point, &other_point);
                let other_label = self.labels[j];

                if other_label == cluster_label {
                    same_cluster_distances.push(distance);
                } else if other_label < self.n_clusters {
                    other_cluster_distances[other_label].push(distance);
                }
            }

            let a_i = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };

            // Calculate b(i): min mean distance to points in other clusters
            let b_i = other_cluster_distances
                .iter()
                .filter(|distances| !distances.is_empty())
                .map(|distances| distances.iter().sum::<f64>() / distances.len() as f64)
                .fold(f64::INFINITY, f64::min);

            let silhouette_i = if b_i.is_infinite() || (a_i == 0.0 && b_i == 0.0) {
                0.0
            } else {
                (b_i - a_i) / a_i.max(b_i)
            };

            silhouette_sum += silhouette_i;
        }

        silhouette_sum / n_samples as f64
    }
}

/// Fit K-Means model on RFM data
///
/// # Arguments
/// * `rfm_data` - Processed RFM data with normalized features
/// * `n_clusters` - Number of clusters (3-5 recommended)
/// * `max_iters` - Maximum iterations for convergence
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// * Fitted `KMeansModel` with predictions and metrics
pub fn fit_kmeans(
    rfm_data: &RfmData,
    n_clusters: usize,
    max_iters: usize,
    tolerance: f64,
) -> crate::Result<KMeansModel> {
    if !(3..=5).contains(&n_clusters) {
        anyhow::bail!(
            "Number of clusters should be between 3 and 5 for meaningful customer segmentation"
        );
    }

    if rfm_data.features.nrows() < n_clusters {
        anyhow::bail!(
            "Number of data points ({}) must be at least equal to number of clusters ({})",
            rfm_data.features.nrows(),
            n_clusters
        );
    }

    // Create dataset for linfa
    let n_samples = rfm_data.features.nrows();
    let targets: Array1<usize> = Array1::zeros(n_samples); // Dummy targets for unsupervised learning
    let dataset = Dataset::new(rfm_data.features.clone(), targets);

    // Configure and fit K-Means
    let model = KMeans::params_with(n_clusters, rand::thread_rng(), L2Dist)
        .max_n_iterations(max_iters as u64)
        .tolerance(tolerance)
        .fit(&dataset)?;

    // Get predictions and centroids
    let labels = model.predict(&dataset);
    let centroids = model.centroids().clone();

    // Compute inertia (within-cluster sum of squares)
    let inertia = compute_inertia(&rfm_data.features, &labels, &centroids);

    Ok(KMeansModel {
        model,
        n_clusters,
        labels,
        centroids,
        inertia,
    })
}

/// Predict cluster for new RFM values
///
/// # Arguments
/// * `model` - Fitted K-Means model
/// * `rfm_data` - Original RFM data (for scaler)
/// * `rfm_values` - New RFM values [recency, frequency, monetary]
///
/// # Returns
/// * Predicted cluster index
pub fn predict_cluster(
    model: &KMeansModel,
    rfm_data: &RfmData,
    rfm_values: &[f64; 3],
) -> crate::Result<usize> {
    let scaled_features = rfm_data.scale_new_data(rfm_values)?;
    model.predict(&scaled_features)
}

/// Compute within-cluster sum of squares (inertia)
fn compute_inertia(features: &Array2<f64>, labels: &Array1<usize>, centroids: &Array2<f64>) -> f64 {
    let mut inertia = 0.0;

    for (i, &cluster) in labels.iter().enumerate() {
        if cluster < centroids.nrows() {
            let point = features.row(i);
            let centroid = centroids.row(cluster);
            let distance_sq = point
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            inertia += distance_sq;
        }
    }

    inertia
}

/// Calculate Euclidean distance between two points
fn euclidean_distance(point1: &ndarray::ArrayView1<f64>, point2: &ndarray::ArrayView1<f64>) -> f64 {
    point1
        .iter()
        .zip(point2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{RfmData, StandardScaler};
    use ndarray::Array2;

    fn create_test_rfm_data() -> RfmData {
        // Create sample normalized features (4 samples, 3 features)
        let features = Array2::from_shape_vec(
            (4, 3),
            vec![
                -1.0, -1.0, -1.0, // Low R, F, M
                1.0, 1.0, 1.0, // High R, F, M
                -0.5, 0.5, -0.5, // Medium values
                0.5, -0.5, 0.5, // Mixed values
            ],
        )
        .unwrap();

        let raw_features = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 1.0, 100.0, 30.0, 10.0, 1000.0, 10.0, 5.0, 500.0, 20.0, 3.0, 750.0,
            ],
        )
        .unwrap();

        let scaler = StandardScaler::fit(&raw_features);

        RfmData {
            features,
            customer_ids: vec![1, 2, 3, 4],
            scaler,
            raw_features,
        }
    }

    #[test]
    fn test_fit_kmeans() {
        let rfm_data = create_test_rfm_data();
        let result = fit_kmeans(&rfm_data, 3, 100, 1e-4);

        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.n_clusters, 3);
        assert_eq!(model.labels.len(), 4);
        assert_eq!(model.centroids.shape(), &[3, 3]);
    }

    #[test]
    fn test_predict_cluster() {
        let rfm_data = create_test_rfm_data();
        let model = fit_kmeans(&rfm_data, 3, 100, 1e-4).unwrap();

        let new_rfm = [15.0, 5.0, 600.0];
        let result = predict_cluster(&model, &rfm_data, &new_rfm);

        assert!(result.is_ok());
        let cluster = result.unwrap();
        assert!(cluster < 3);
    }

    #[test]
    fn test_cluster_sizes() {
        let rfm_data = create_test_rfm_data();
        let model = fit_kmeans(&rfm_data, 3, 100, 1e-4).unwrap();

        let sizes = model.cluster_sizes();
        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes.iter().sum::<usize>(), 4); // Total should equal number of samples
    }

    #[test]
    fn test_invalid_cluster_count() {
        let rfm_data = create_test_rfm_data();

        // Too few clusters
        let result = fit_kmeans(&rfm_data, 2, 100, 1e-4);
        assert!(result.is_err());

        // Too many clusters
        let result = fit_kmeans(&rfm_data, 6, 100, 1e-4);
        assert!(result.is_err());
    }
}
