//! ClusterForge: Customer Segmentation CLI using K-Means clustering on RFM analysis
//!
//! This is the main entrypoint that orchestrates data loading, model fitting,
//! visualization, and prediction.

use anyhow::Result;
use clap::Parser;
use clusterforge::{Args, fit_kmeans, load_and_process_data, predict_cluster, viz};
use std::time::Instant;

fn main() -> Result<()> {
    // Parse command-line arguments
    let args = Args::parse();

    if args.verbose {
        println!("ClusterForge - Customer Segmentation using K-Means");
        println!("================================================\n");
    }

    // Check if in prediction mode
    if let Some(rfm_values) = args.parse_rfm_values()? {
        run_prediction_mode(&args, rfm_values)?;
    } else {
        run_full_pipeline(&args)?;
    }

    Ok(())
}

/// Run prediction mode for a single customer
fn run_prediction_mode(args: &Args, rfm_values: (f64, f64, f64)) -> Result<()> {
    println!("=== Prediction Mode ===");
    println!(
        "Input RFM values: R={}, F={}, M={}",
        rfm_values.0, rfm_values.1, rfm_values.2
    );

    let start_time = Instant::now();

    // Load and process data to fit the model
    if args.verbose {
        println!("\nLoading training data from: {}", args.input);
    }
    let rfm_data = load_and_process_data(&args.input, None)?;

    if args.verbose {
        println!("Loaded {} customers", rfm_data.customer_ids.len());
        println!("\nFitting K-Means model with {} clusters...", args.clusters);
    }

    // Fit model
    let model = fit_kmeans(&rfm_data, args.clusters, args.max_iters, args.tolerance)?;

    // Predict cluster for new data
    let rfm_array = [rfm_values.0, rfm_values.1, rfm_values.2];
    let cluster = predict_cluster(&model, &rfm_data, &rfm_array)?;

    let elapsed = start_time.elapsed();

    println!("\n✓ Predicted Cluster: {}", cluster);
    println!("  Processing time: {:.2}s", elapsed.as_secs_f64());

    // Show cluster context
    let cluster_sizes = model.cluster_sizes();
    let total_customers = rfm_data.customer_ids.len();
    let cluster_percentage = (cluster_sizes[cluster] as f64 / total_customers as f64) * 100.0;

    println!("\nCluster {} details:", cluster);
    println!(
        "  Size: {} customers ({:.1}% of total)",
        cluster_sizes[cluster], cluster_percentage
    );
    println!(
        "  Centroid (normalized): R={:.2}, F={:.2}, M={:.2}",
        model.centroids[[cluster, 0]],
        model.centroids[[cluster, 1]],
        model.centroids[[cluster, 2]]
    );

    Ok(())
}

/// Run full clustering pipeline
fn run_full_pipeline(args: &Args) -> Result<()> {
    println!("=== Full Clustering Pipeline ===\n");

    let start_time = Instant::now();

    // Step 1: Load and process data
    if args.verbose {
        println!("Step 1: Loading and processing data");
        println!("  Input file: {}", args.input);
    }

    let data_start = Instant::now();
    let rfm_data = load_and_process_data(&args.input, None)?;
    let data_time = data_start.elapsed();

    println!("✓ Data loaded: {} customers", rfm_data.customer_ids.len());
    if args.verbose {
        println!("  Processing time: {:.2}s", data_time.as_secs_f64());
        println!("  Features shape: {:?}", rfm_data.features.shape());
    }

    // Step 2: Fit K-Means model
    if args.verbose {
        println!("\nStep 2: Fitting K-Means model");
        println!("  Number of clusters: {}", args.clusters);
        println!("  Max iterations: {}", args.max_iters);
        println!("  Tolerance: {}", args.tolerance);
    }

    let model_start = Instant::now();
    let model = fit_kmeans(&rfm_data, args.clusters, args.max_iters, args.tolerance)?;
    let model_time = model_start.elapsed();

    println!("✓ Model fitted successfully");
    if args.verbose {
        println!("  Fitting time: {:.2}s", model_time.as_secs_f64());
        println!("  Inertia: {:.2}", model.inertia);
    }

    // Step 3: Print cluster statistics
    println!("\n=== Cluster Statistics ===");
    let cluster_sizes = model.cluster_sizes();
    for (i, &size) in cluster_sizes.iter().enumerate() {
        let percentage = (size as f64 / rfm_data.customer_ids.len() as f64) * 100.0;
        println!("Cluster {}: {} customers ({:.1}%)", i, size, percentage);
    }

    // Compute silhouette score on a sample
    let silhouette_score =
        model.compute_silhouette_sample(&rfm_data.features, 100.min(rfm_data.customer_ids.len()));
    println!("\nSilhouette score (sample): {:.3}", silhouette_score);
    println!("Within-cluster sum of squares: {:.2}", model.inertia);

    // Step 4: Generate visualization
    if args.verbose {
        println!("\nStep 3: Generating visualizations");
        println!("  Output file: {}", args.output);
    }

    let viz_start = Instant::now();
    viz::generate_visualization_report(&rfm_data, &model, &args.output)?;
    let viz_time = viz_start.elapsed();

    println!("\n✓ Visualizations generated");
    if args.verbose {
        println!("  Visualization time: {:.2}s", viz_time.as_secs_f64());
    }

    let total_time = start_time.elapsed();
    println!("\n=== Pipeline Complete ===");
    println!("Total processing time: {:.2}s", total_time.as_secs_f64());
    println!("Main plot saved to: {}", args.output);
    println!(
        "Cluster sizes saved to: {}",
        args.output.replace(".png", "_sizes.png")
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_compiles() {
        // Basic compilation test
        assert!(true);
    }
}
