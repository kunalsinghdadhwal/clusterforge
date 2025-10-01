//! ClusterForge: A Rust CLI application for customer segmentation using K-Means clustering
//! 
//! This library provides functionality for RFM (Recency, Frequency, Monetary) analysis
//! on customer transaction data using K-Means clustering.

pub mod cli;
pub mod data;
pub mod model;
pub mod viz;

// Re-export public items for easier access
pub use cli::Args;
pub use data::{load_and_process_data, RfmData};
pub use model::{KMeansModel, fit_kmeans, predict_cluster};
pub use viz::create_cluster_visualization;

/// Common result type used throughout the application
pub type Result<T> = anyhow::Result<T>;
