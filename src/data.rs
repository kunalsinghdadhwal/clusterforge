//! Data loading and RFM feature computation using Polars

use chrono::{DateTime, Utc};
use ndarray::{Array2, Array1};
use polars::prelude::*;

/// RFM data structure containing processed features and scaler
#[derive(Debug)]
pub struct RfmData {
    /// Normalized RFM features as ndarray (n_customers, 3)
    pub features: Array2<f64>,
    /// Customer IDs corresponding to each row
    pub customer_ids: Vec<i64>,
    /// Fitted StandardScaler for normalizing new data
    pub scaler: StandardScaler<f64>,
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
        let scaled = self.scaler.transform(input);
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

    // Load data using Polars lazy frame for efficiency
    let df = LazyFrame::scan_csv(file_path, ScanArgsCSV::default())?
        .filter(
            // Filter out invalid rows
            col("Quantity").gt(0)
                .and(col("UnitPrice").gt(0.0))
                .and(col("CustomerID").is_not_null())
        )
        .with_columns([
            // Parse InvoiceDate and add TotalAmount
            col("InvoiceDate").str().strptime(PolarsDataType::Datetime(TimeUnit::Microseconds, None), 
                StrptimeOptions::default()),
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
            // Frequency: number of unique invoices
            col("InvoiceNo").n_unique().alias("Frequency"),
            // Monetary: total spending
            col("TotalAmount").sum().alias("Monetary")
        ])
        .with_columns([
            // Calculate recency in days
            ((lit(reference_timestamp) - col("LastPurchaseDate")) / 1_000_000 / 86400)
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
fn prepare_features(df: DataFrame) -> crate::Result<(Array2<f64>, Vec<i64>, StandardScaler<f64>, Array2<f64>)> {
    // Extract customer IDs
    let customer_ids: Vec<i64> = df.column("CustomerID")?
        .i64()?
        .into_no_null_iter()
        .collect();

    // Extract RFM features
    let recency: Vec<f64> = df.column("Recency")?
        .f64()?
        .into_no_null_iter()
        .collect();
    
    let frequency: Vec<f64> = df.column("Frequency")?
        .cast(&DataType::Float64)?
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
    let dataset = Dataset::new(raw_features.clone(), Array1::zeros(n_samples));
    let scaler = StandardScaler::default().fit(&dataset)?;
    
    // Transform features
    let normalized_features = scaler.transform(raw_features.clone());
    
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
        writeln!(file, "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01T08:26:00Z,2.55,17850,United Kingdom").unwrap();
        writeln!(file, "536365,71053,WHITE METAL LANTERN,6,2010-12-01T08:26:00Z,3.39,17850,United Kingdom").unwrap();
        writeln!(file, "536366,22633,HAND WARMER UNION JACK,6,2010-12-01T08:28:00Z,1.85,17850,United Kingdom").unwrap();
        writeln!(file, "536367,84406B,CREAM CUPID HEARTS COAT HANGER,8,2010-12-01T08:34:00Z,2.75,13047,United Kingdom").unwrap();
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
}
