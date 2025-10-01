//! Visualization functions using Plotters for cluster analysis

use plotters::prelude::*;
use crate::model::KMeansModel;
use crate::data::RfmData;

/// Color palette for different clusters
const CLUSTER_COLORS: [RGBColor; 5] = [
    RED,
    BLUE, 
    GREEN,
    YELLOW,
    MAGENTA,
];

/// Create scatter plot visualization of clusters
/// 
/// # Arguments
/// * `rfm_data` - Original RFM data
/// * `model` - Fitted K-Means model with cluster assignments
/// * `output_path` - Path to save the PNG plot
/// * `plot_title` - Title for the plot
/// 
/// # Returns
/// * Result indicating success or failure
pub fn create_cluster_visualization(
    rfm_data: &RfmData,
    model: &KMeansModel,
    output_path: &str,
    plot_title: Option<&str>,
) -> crate::Result<()> {
    let title = plot_title.unwrap_or("Customer Segmentation: Frequency vs Monetary (Colored by Cluster)");
    
    // Use normalized features for consistent scaling
    let features = &rfm_data.features;
    let labels = &model.labels;
    
    // Extract Frequency (index 1) and Monetary (index 2) values
    let frequency_values: Vec<f64> = features.column(1).to_vec();
    let monetary_values: Vec<f64> = features.column(2).to_vec();
    
    // Calculate plot bounds with some padding
    let freq_min = frequency_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 0.5;
    let freq_max = frequency_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 0.5;
    let mon_min = monetary_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 0.5;
    let mon_max = monetary_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 0.5;
    
    // Create the drawing backend
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(freq_min..freq_max, mon_min..mon_max)?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Normalized)")
        .y_desc("Monetary (Normalized)")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Plot data points colored by cluster
    for (i, (&freq, &mon)) in frequency_values.iter().zip(monetary_values.iter()).enumerate() {
        let cluster = labels[i];
        let color = if cluster < CLUSTER_COLORS.len() {
            &CLUSTER_COLORS[cluster]
        } else {
            &BLACK // Fallback color
        };
        
        chart.draw_series(std::iter::once(Circle::new((freq, mon), 4, color.filled())))?;
    }

    // Plot centroids as larger squares
    let centroids = &model.centroids;
    for (cluster_id, centroid_row) in centroids.outer_iter().enumerate() {
        let freq_centroid = centroid_row[1];
        let mon_centroid = centroid_row[2];
        let color = if cluster_id < CLUSTER_COLORS.len() {
            &CLUSTER_COLORS[cluster_id]
        } else {
            &BLACK
        };
        
        chart.draw_series(std::iter::once(
            Rectangle::new([(freq_centroid - 0.1, mon_centroid - 0.1), 
                           (freq_centroid + 0.1, mon_centroid + 0.1)], 
                          color.filled())
        ))?
        .label(format!("Cluster {} Centroid", cluster_id))
        .legend(move |(x, y)| Rectangle::new([(x, y), (x + 10, y + 10)], color.filled()));
    }

    chart.configure_series_labels().draw()?;
    
    root.present()?;
    println!("Cluster visualization saved to: {}", output_path);
    
    Ok(())
}

/// Create a simple histogram of cluster sizes
pub fn create_cluster_size_chart(
    model: &KMeansModel,
    output_path: &str,
) -> crate::Result<()> {
    let cluster_sizes = model.cluster_sizes();
    let max_size = *cluster_sizes.iter().max().unwrap_or(&1) as f64;
    
    let root = BitMapBackend::new(output_path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Cluster Sizes", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..(model.n_clusters as f64), 0f64..(max_size * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Cluster ID")
        .y_desc("Number of Customers")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Draw bars for each cluster
    for (cluster_id, &size) in cluster_sizes.iter().enumerate() {
        let color = if cluster_id < CLUSTER_COLORS.len() {
            &CLUSTER_COLORS[cluster_id]
        } else {
            &BLUE
        };
        
        chart.draw_series(std::iter::once(
            Rectangle::new([(cluster_id as f64 - 0.4, 0.0), 
                           (cluster_id as f64 + 0.4, size as f64)], 
                          color.filled())
        ))?;
    }
    
    root.present()?;
    println!("Cluster size chart saved to: {}", output_path);
    
    Ok(())
}

/// Print cluster statistics to console
pub fn print_cluster_statistics(rfm_data: &RfmData, model: &KMeansModel) {
    println!("\n=== Cluster Statistics ===");
    println!("Number of clusters: {}", model.n_clusters);
    println!("Total customers: {}", rfm_data.customer_ids.len());
    println!("Within-cluster sum of squares (Inertia): {:.2}", model.inertia);
    
    // Basic silhouette score on a sample
    let silhouette_score = model.compute_silhouette_sample(&rfm_data.features, 100);
    println!("Silhouette score (sample): {:.3}", silhouette_score);
    
    let cluster_sizes = model.cluster_sizes();
    println!("\nCluster sizes:");
    for (i, &size) in cluster_sizes.iter().enumerate() {
        let percentage = (size as f64 / rfm_data.customer_ids.len() as f64) * 100.0;
        println!("  Cluster {}: {} customers ({:.1}%)", i, size, percentage);
    }
    
    // Print centroid information (in normalized space)
    println!("\nCluster centroids (normalized):");
    println!("  Cluster | Recency | Frequency | Monetary");
    println!("  --------|---------|-----------|----------");
    for (i, centroid_row) in model.centroids.outer_iter().enumerate() {
        println!("  {:7} | {:7.2} | {:9.2} | {:8.2}", 
                i, centroid_row[0], centroid_row[1], centroid_row[2]);
    }
}

/// Generate a comprehensive visualization report
pub fn generate_visualization_report(
    rfm_data: &RfmData,
    model: &KMeansModel,
    base_output_path: &str,
) -> crate::Result<()> {
    // Main cluster plot
    let main_plot_path = base_output_path;
    create_cluster_visualization(rfm_data, model, main_plot_path, None)?;
    
    // Cluster size chart
    let size_chart_path = base_output_path.replace(".png", "_sizes.png");
    create_cluster_size_chart(model, &size_chart_path)?;
    
    // Print statistics
    print_cluster_statistics(rfm_data, model);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::path::Path;
    use crate::data::{RfmData, StandardScaler};
    use linfa::{Dataset, prelude::*};
    use linfa_clustering::KMeans;
    use linfa_nn::distance::L2Dist;
    use ndarray::{Array1, Array2};

    fn create_test_data() -> (RfmData, KMeansModel) {
        // Create test RFM data
        let features = Array2::from_shape_vec((6, 3), vec![
            -1.0, -1.0, -1.0,
             1.0,  1.0,  1.0,
            -0.5,  0.5, -0.5,
             0.5, -0.5,  0.5,
             0.0,  0.0,  0.0,
            -0.2,  0.8, -0.8,
        ]).unwrap();
        
        let raw_features = features.clone();
        let scaler = StandardScaler::fit(&raw_features);
        
        let rfm_data = RfmData {
            features: features.clone(),
            customer_ids: vec![1, 2, 3, 4, 5, 6],
            scaler,
            raw_features,
        };
        
        // Create K-Means model
        let dataset: Dataset<f64, usize, ndarray::Dim<[usize; 1]>> = Dataset::new(features, Array1::zeros(6));
        let linfa_model = KMeans::params_with(3, rand::thread_rng(), L2Dist)
            .fit(&dataset)
            .unwrap();
        
        let labels = linfa_model.predict(&dataset);
        let centroids = linfa_model.centroids().clone();
        
        let model = KMeansModel {
            model: linfa_model,
            n_clusters: 3,
            labels,
            centroids,
            inertia: 2.5,
        };
        
        (rfm_data, model)
    }

    #[test]
    fn test_create_cluster_visualization() {
        let (rfm_data, model) = create_test_data();
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test_plot.png");
        let output_str = output_path.to_str().unwrap();
        
        let result = create_cluster_visualization(&rfm_data, &model, output_str, None);
        assert!(result.is_ok());
        assert!(Path::new(output_str).exists());
    }

    #[test]
    fn test_create_cluster_size_chart() {
        let (_rfm_data, model) = create_test_data();
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test_sizes.png");
        let output_str = output_path.to_str().unwrap();
        
        let result = create_cluster_size_chart(&model, output_str);
        assert!(result.is_ok());
        assert!(Path::new(output_str).exists());
    }

    #[test]
    fn test_generate_visualization_report() {
        let (rfm_data, model) = create_test_data();
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test_report.png");
        let output_str = output_path.to_str().unwrap();
        
        let result = generate_visualization_report(&rfm_data, &model, output_str);
        assert!(result.is_ok());
        assert!(Path::new(output_str).exists());
    }
}
