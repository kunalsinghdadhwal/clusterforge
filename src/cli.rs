//! Command-line interface definitions and argument parsing

use clap::Parser;

/// Customer segmentation CLI using K-Means clustering on RFM data
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the input CSV file
    #[arg(short, long, default_value = "data.csv")]
    pub input: String,

    /// Number of clusters for K-Means
    #[arg(short = 'k', long, default_value = "4")]
    pub clusters: usize,

    /// Output path for the visualization plot
    #[arg(short, long, default_value = "cluster_plot.png")]
    pub output: String,

    /// Prediction mode: provide R,F,M values as comma-separated string
    /// Example: --predict "30,10,500.0" for Recency=30, Frequency=10, Monetary=500.0
    #[arg(short, long)]
    pub predict: Option<String>,

    /// Maximum iterations for K-Means algorithm
    #[arg(long, default_value = "300")]
    pub max_iters: usize,

    /// Tolerance for K-Means convergence
    #[arg(long, default_value = "1e-4")]
    pub tolerance: f64,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

impl Args {
    /// Parse RFM values from the predict string
    /// Expected format: "recency,frequency,monetary"
    pub fn parse_rfm_values(&self) -> crate::Result<Option<(f64, f64, f64)>> {
        if let Some(ref predict_str) = self.predict {
            let parts: Vec<&str> = predict_str.split(',').collect();
            if parts.len() != 3 {
                anyhow::bail!("Predict values must be in format 'recency,frequency,monetary'");
            }

            let recency: f64 = parts[0]
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid recency value: {}", parts[0]))?;
            let frequency: f64 = parts[1]
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid frequency value: {}", parts[1]))?;
            let monetary: f64 = parts[2]
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid monetary value: {}", parts[2]))?;

            Ok(Some((recency, frequency, monetary)))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rfm_values() {
        let mut args = Args {
            input: "test.csv".to_string(),
            clusters: 4,
            output: "test.png".to_string(),
            predict: Some("30,10,500.0".to_string()),
            max_iters: 300,
            tolerance: 1e-4,
            verbose: false,
        };

        let result = args.parse_rfm_values().unwrap();
        assert_eq!(result, Some((30.0, 10.0, 500.0)));

        args.predict = None;
        let result = args.parse_rfm_values().unwrap();
        assert_eq!(result, None);

        args.predict = Some("invalid".to_string());
        assert!(args.parse_rfm_values().is_err());
    }
}
