//! Integration tests for ClusterForge

use clusterforge::{fit_kmeans, load_and_process_data, predict_cluster};
use std::io::Write;
use tempfile::NamedTempFile;

/// Create a test CSV file with sample data
fn create_test_csv() -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country"
    )
    .unwrap();

    // Customer 17850 - multiple purchases
    writeln!(file, "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01T08:26:00,2.55,17850,United Kingdom").unwrap();
    writeln!(
        file,
        "536365,71053,WHITE METAL LANTERN,6,2010-12-01T08:26:00,3.39,17850,United Kingdom"
    )
    .unwrap();
    writeln!(
        file,
        "536366,22633,HAND WARMER UNION JACK,6,2011-11-01T08:28:00,1.85,17850,United Kingdom"
    )
    .unwrap();

    // Customer 13047 - single purchase
    writeln!(file, "536367,84406B,CREAM CUPID HEARTS COAT HANGER,8,2010-12-01T08:34:00,2.75,13047,United Kingdom").unwrap();

    // Customer 12345 - recent high value
    writeln!(
        file,
        "536368,22752,SET 7 BABUSHKA NESTING BOXES,2,2011-12-05T10:15:00,7.65,12345,United Kingdom"
    )
    .unwrap();
    writeln!(file, "536368,21730,GLASS STAR FROSTED T-LIGHT HOLDER,12,2011-12-05T10:15:00,1.25,12345,United Kingdom").unwrap();

    // Customer 98765 - old low value
    writeln!(file, "536369,22457,NATURAL SLATE HEART CHALKBOARD,4,2010-01-15T09:00:00,3.25,98765,United Kingdom").unwrap();

    file
}

#[test]
fn test_end_to_end_pipeline() {
    // Create test data
    let test_file = create_test_csv();
    let file_path = test_file.path().to_str().unwrap();

    // Load and process data
    let rfm_data = load_and_process_data(file_path, Some("2011-12-09T00:00:00Z")).unwrap();

    // Verify data loading
    assert_eq!(rfm_data.customer_ids.len(), 4); // 4 unique customers
    assert_eq!(rfm_data.features.shape(), &[4, 3]); // 4 customers, 3 features

    // Fit K-Means model
    let model = fit_kmeans(&rfm_data, 3, 100, 1e-4).unwrap();

    // Verify model
    assert_eq!(model.n_clusters, 3);
    assert_eq!(model.labels.len(), 4);
    assert_eq!(model.centroids.shape(), &[3, 3]);

    // Verify all customers are assigned to a cluster
    for &label in model.labels.iter() {
        assert!(label < 3);
    }

    // Verify cluster sizes sum to total customers
    let cluster_sizes = model.cluster_sizes();
    let total: usize = cluster_sizes.iter().sum();
    assert_eq!(total, 4);
}

#[test]
fn test_prediction() {
    let test_file = create_test_csv();
    let file_path = test_file.path().to_str().unwrap();

    // Load data and fit model
    let rfm_data = load_and_process_data(file_path, Some("2011-12-09T00:00:00Z")).unwrap();
    let model = fit_kmeans(&rfm_data, 3, 100, 1e-4).unwrap();

    // Predict cluster for new customer
    let new_rfm = [10.0, 5.0, 250.0]; // Medium recency, frequency, monetary
    let cluster = predict_cluster(&model, &rfm_data, &new_rfm).unwrap();

    // Verify prediction is valid
    assert!(cluster < 3);
}

#[test]
fn test_error_handling_invalid_clusters() {
    let test_file = create_test_csv();
    let file_path = test_file.path().to_str().unwrap();

    let rfm_data = load_and_process_data(file_path, Some("2011-12-09T00:00:00Z")).unwrap();

    // Try with invalid number of clusters (< 3)
    let result = fit_kmeans(&rfm_data, 2, 100, 1e-4);
    assert!(result.is_err());

    // Try with invalid number of clusters (> 5)
    let result = fit_kmeans(&rfm_data, 6, 100, 1e-4);
    assert!(result.is_err());
}

#[test]
fn test_rfm_computation() {
    let test_file = create_test_csv();
    let file_path = test_file.path().to_str().unwrap();

    let rfm_data = load_and_process_data(file_path, Some("2011-12-09T00:00:00Z")).unwrap();

    // Verify RFM features are normalized (mean ~0, std ~1)
    // Check that features are in reasonable range after normalization
    for row in rfm_data.features.outer_iter() {
        for &value in row.iter() {
            // Normalized values should typically be in range [-3, 3]
            assert!(
                value.abs() < 10.0,
                "Normalized value {} is out of expected range",
                value
            );
        }
    }

    // Verify raw features have reasonable values
    assert!(rfm_data.raw_features.iter().all(|&x| x >= 0.0));
}

#[test]
fn test_model_inertia() {
    let test_file = create_test_csv();
    let file_path = test_file.path().to_str().unwrap();

    let rfm_data = load_and_process_data(file_path, Some("2011-12-09T00:00:00Z")).unwrap();
    let model = fit_kmeans(&rfm_data, 3, 100, 1e-4).unwrap();

    // Inertia should be non-negative
    assert!(model.inertia >= 0.0);

    // Inertia should be finite
    assert!(model.inertia.is_finite());
}
