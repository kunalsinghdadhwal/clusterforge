//! Data loading and RFM feature computation using Polars

use chrono::{DateTime, Utc};
use ndarray::{Array2, Array1};
use polars::prelude::*;

/// Simple StandardScaler implementation for feature normalization
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl StandardScaler {
    /// Fit the scaler to data
    pub fn fit(data: &Array2<f64>) -> Self {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        
        let mut mean = Array1::zeros(n_features);
        let mut std = Array1::zeros(n_features);
        
        // Calculate mean
        for col_idx in 0..n_features {
            let col_sum: f64 = data.column(col_idx).iter().sum();
            mean[col_idx] = col_sum / n_samples as f64;
        }
        
        // Calculate standard deviation
        for col_idx in 0..n_features {
            let variance: f64 = data.column(col_idx)
                .iter()
                .map(|&x| (x - mean[col_idx]).powi(2))
                .sum::<f64>() / n_samples as f64;
            std[col_idx] = variance.sqrt().max(1e-8); // Avoid division by zero
        }
        
        StandardScaler { mean, std }
    }
    
    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut normalized = data.clone();
        for col_idx in 0..data.ncols() {
            for row_idx in 0..data.nrows() {
                normalized[[row_idx, col_idx]] = 
                    (data[[row_idx, col_idx]] - self.mean[col_idx]) / self.std[col_idx];
            }
        }
        normalized
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(data: &Array2<f64>) -> (Self, Array2<f64>) {
        let scaler = Self::fit(data);
        let transformed = scaler.transform(data);
        (scaler, transformed)
    }
}

/// RFM data structure containing processed features and scaler
#[derive(Debug)]
pub struct RfmData {
    /// Normalized RFM features as ndarray (n_customers, 3)
    pub features: Array2<f64>,
    /// Customer IDs corresponding to each row
    pub customer_ids: Vec<i64>,
    /// Fitted StandardScaler for normalizing new data
    pub scaler: StandardScaler,
    /// Raw RFM values before normalization
    pub raw_features: Array2<f64>,
}

impl RfmData {
    /// Scale new RFM values using the fitted scaler
    pub fn scale_new_data(&self, rfm: &[f64; 3]) -> crate::Result<Array1<f64>> {
        if rfm.len() != 3 {
            anyhow::bail!("RFM data must have exactly 3 features");
        }
        
        let input = Array2::from_shape_vec((1, 3), rfm.to_vec())?;
        let scaled = self.scaler.transform(&input);
        Ok(scaled.row(0).to_owned())
    }
}

/// Load CSV data and compute RFM features with normalization
/// 
/// # Arguments
/// * `file_path` - Path to the CSV file
/// * `end_date` - Reference date for recency calculation (default: 2011-12-09)
/// 
/// # Returns
/// * `RfmData` containing normalized features and metadata
pub fn load_and_process_data(file_path: &str, end_date: Option<&str>) -> crate::Result<RfmData> {
    let end_date_str = end_date.unwrap_or("2011-12-09T00:00:00Z");
    let reference_date = DateTime::parse_from_rfc3339(end_date_str)?
        .with_timezone(&Utc);

    // Load data using Polars with proper schema inference
    let df = LazyCsvReader::new(file_path)
        .with_infer_schema_length(Some(10000))
        .with_ignore_errors(true)
        .finish()?
        .filter(
            // Filter out invalid rows (cancellations start with 'C')
            col("Quantity").gt(0)
                .and(col("UnitPrice").gt(0.0))
                .and(col("CustomerID").is_not_null())
        )
        .with_columns([
            // Parse InvoiceDate with the correct format: MM/DD/YYYY HH:MM:SS
            col("InvoiceDate").str().to_datetime(
                Some(TimeUnit::Microseconds),
                None,
                StrptimeOptions {
                    format: Some("%m/%d/%Y %H:%M:%S".to_string()),
                    ..Default::default()
                },
                lit("raise")
            ),
            (col("Quantity") * col("UnitPrice")).alias("TotalAmount")
        ])
        .collect()?;

    if df.height() == 0 {
        anyhow::bail!("No valid data found after filtering");
    }

    // Compute RFM features
    let rfm_df = compute_rfm_features(df, reference_date)?;
    
    // Convert to ndarray and normalize
    let (features, customer_ids, scaler, raw_features) = prepare_features(rfm_df)?;
    
    Ok(RfmData {
        features,
        customer_ids,
        scaler,
        raw_features,
    })
}

/// Compute RFM features from transaction data
fn compute_rfm_features(df: DataFrame, reference_date: DateTime<Utc>) -> crate::Result<DataFrame> {
    // Convert reference date to microseconds for Polars comparison
    let reference_timestamp = reference_date.timestamp_micros();
    
    let rfm_df = df
        .lazy()
        .group_by([col("CustomerID")])
        .agg([
            // Recency: days since last purchase
            col("InvoiceDate").max().alias("LastPurchaseDate"),
            // Frequency: number of unique invoices (cast to f64)
            col("InvoiceNo").n_unique().cast(DataType::Float64).alias("Frequency"),
            // Monetary: total spending
            col("TotalAmount").sum().alias("Monetary")
        ])
        .with_columns([
            // Calculate recency in days
            ((lit(reference_timestamp) - col("LastPurchaseDate")) / lit(1_000_000) / lit(86400))
                .cast(DataType::Float64)
                .alias("Recency")
        ])
        .select([
            col("CustomerID"),
            col("Recency"),
            col("Frequency"),
            col("Monetary")
        ])
        .filter(
            // Additional filters for data quality
            col("Recency").gt_eq(0.0)
                .and(col("Frequency").gt(0))
                .and(col("Monetary").gt(0.0))
        )
        .collect()?;

    if rfm_df.height() == 0 {
        anyhow::bail!("No customers found after RFM computation");
    }

    Ok(rfm_df)
}

/// Convert DataFrame to ndarray and apply StandardScaler normalization
fn prepare_features(df: DataFrame) -> crate::Result<(Array2<f64>, Vec<i64>, StandardScaler, Array2<f64>)> {
    // Extract customer IDs
    let customer_ids: Vec<i64> = df.column("CustomerID")?
        .i64()?
        .into_no_null_iter()
        .collect();

    // Extract RFM features (all should be Float64 now)
    let recency: Vec<f64> = df.column("Recency")?
        .f64()?
        .into_no_null_iter()
        .collect();
    
    let frequency: Vec<f64> = df.column("Frequency")?
        .f64()?
        .into_no_null_iter()
        .collect();
    
    let monetary: Vec<f64> = df.column("Monetary")?
        .f64()?
        .into_no_null_iter()
        .collect();

    let n_samples = customer_ids.len();
    
    // Create raw feature matrix
    let mut raw_data = Vec::with_capacity(n_samples * 3);
    for i in 0..n_samples {
        raw_data.extend_from_slice(&[recency[i], frequency[i], monetary[i]]);
    }
    
    let raw_features = Array2::from_shape_vec((n_samples, 3), raw_data.clone())?;
    
    // Create and fit StandardScaler
    let (scaler, normalized_features) = StandardScaler::fit_transform(&raw_features);
    
    Ok((normalized_features, customer_ids, scaler, raw_features))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    fn create_test_csv() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country").unwrap();
        writeln!(file, "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/01/2010 08:26:00,2.55,17850,United Kingdom").unwrap();
        writeln!(file, "536365,71053,WHITE METAL LANTERN,6,12/01/2010 08:26:00,3.39,17850,United Kingdom").unwrap();
        writeln!(file, "536366,22633,HAND WARMER UNION JACK,6,12/01/2010 08:28:00,1.85,17850,United Kingdom").unwrap();
        writeln!(file, "536367,84406B,CREAM CUPID HEARTS COAT HANGER,8,12/01/2010 08:34:00,2.75,13047,United Kingdom").unwrap();
        writeln!(file, "C536368,22632,HAND WARMER RED POLKA DOT,-6,12/01/2010 08:35:00,1.85,17850,United Kingdom").unwrap(); // Cancellation - should be filtered
        file
    }

    #[test]
    fn test_load_and_process_data() {
        let test_file = create_test_csv();
        let file_path = test_file.path().to_str().unwrap();
        
        let result = load_and_process_data(file_path, Some("2011-12-09T00:00:00Z"));
        assert!(result.is_ok());
        
        let rfm_data = result.unwrap();
        assert_eq!(rfm_data.features.shape(), &[2, 3]); // 2 customers, 3 features
        assert_eq!(rfm_data.customer_ids.len(), 2);
    }

    #[test]
    fn test_scale_new_data() {
        let test_file = create_test_csv();
        let file_path = test_file.path().to_str().unwrap();
        
        let rfm_data = load_and_process_data(file_path, Some("2011-12-09T00:00:00Z")).unwrap();
        let new_rfm = [30.0, 10.0, 500.0];
        
        let result = rfm_data.scale_new_data(&new_rfm);
        assert!(result.is_ok());
        
        let scaled = result.unwrap();
        assert_eq!(scaled.len(), 3);
    }
    
    #[test]
    fn test_standard_scaler() {
        let data = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            4.0, 8.0,
        ]).unwrap();
        
        let scaler = StandardScaler::fit(&data);
        let transformed = scaler.transform(&data);
        
        // Check that mean is close to 0
        let col0_mean: f64 = transformed.column(0).iter().sum::<f64>() / 4.0;
        assert!(col0_mean.abs() < 1e-10);
    }
}
