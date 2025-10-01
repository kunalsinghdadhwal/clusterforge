# ClusterForge

A high-performance Rust CLI application for customer segmentation using K-Means clustering on transactional data. ClusterForge implements RFM (Recency, Frequency, Monetary) analysis to identify distinct customer segments, enabling data-driven marketing strategies and customer relationship management.

## Overview

ClusterForge processes large-scale retail transaction datasets (500k+ rows in under 2 seconds) to segment customers based on their purchasing behavior. The application leverages efficient data structures and parallel processing capabilities of Rust to deliver production-ready customer analytics.

### Key Features

- **RFM Analysis**: Automatic computation of Recency, Frequency, and Monetary features from transaction data
- **K-Means Clustering**: Unsupervised learning to identify 3-5 distinct customer segments
- **Feature Normalization**: StandardScaler implementation for consistent feature scaling
- **Visual Analytics**: Scatter plots and cluster size charts using Plotters
- **Prediction Mode**: Classify new customers into existing segments
- **Performance**: Sub-second processing for datasets with 500k+ transactions
- **Error Handling**: Comprehensive validation with detailed error messages

## Architecture

### Modular Design

The application follows a clean separation of concerns with distinct modules:

```
src/
├── main.rs          # CLI orchestration and pipeline coordination
├── lib.rs           # Public API exports
├── cli.rs           # Command-line argument parsing (Clap)
├── data.rs          # Data loading, RFM computation, normalization
├── model.rs         # K-Means implementation and prediction
└── viz.rs           # Visualization generation (Plotters)
```

### Data Processing Pipeline

1. **Data Loading**: Polars LazyFrame API for efficient CSV parsing with schema inference
2. **Filtering**: Removal of cancellations (negative quantities), null values, and invalid transactions
3. **RFM Computation**: 
   - Recency: Days since last purchase from reference date
   - Frequency: Count of unique invoices per customer
   - Monetary: Total spending (Quantity × UnitPrice)
4. **Normalization**: Custom StandardScaler implementation (z-score normalization)
5. **Clustering**: Linfa K-Means with configurable parameters
6. **Visualization**: 2D scatter plots (Frequency vs Monetary) colored by cluster

### Technical Implementation

#### RFM Feature Engineering

The application computes customer-level features from transaction-level data:

```rust
// Aggregation in Polars
.group_by([col("CustomerID")])
.agg([
    col("InvoiceDate").max(),           // Last purchase date
    col("InvoiceNo").n_unique(),        // Transaction count
    col("TotalAmount").sum()            // Total spending
])
```

Recency is calculated as the time difference between a reference date (default: 2011-12-09) and the customer's last purchase, converted to days.

#### StandardScaler

Custom implementation avoiding external preprocessing dependencies:

- Computes mean (μ) and standard deviation (σ) per feature
- Transforms features: `z = (x - μ) / σ`
- Prevents division by zero with minimum threshold (1e-8)
- Stores parameters for consistent transformation of new data

#### K-Means Clustering

Uses Linfa's K-Means implementation with L2 (Euclidean) distance:

- Configurable cluster count (3-5 recommended for customer segmentation)
- Maximum iterations: 300 (default)
- Convergence tolerance: 1e-4
- Random initialization with thread-local RNG
- Returns cluster centroids and assignments

#### Performance Optimizations

- **Lazy Evaluation**: Polars LazyFrame defers computation until `.collect()`
- **Zero-Copy Operations**: ndarray views where possible
- **Efficient Aggregations**: Vectorized operations in Polars
- **Release Mode**: Compiled with optimizations enabled

## Installation

### Prerequisites

- Rust 1.70 or later
- Cargo package manager

### Build from Source

```bash
git clone <repository-url>
cd clusterforge
cargo build --release
```

The compiled binary will be available at `target/release/clusterforge`.

## Usage

### Basic Clustering

Perform RFM analysis and generate cluster visualizations:

```bash
cargo run --release -- --input data.csv --clusters 4 --output cluster_plot.png
```

**Output:**
```
=== Full Clustering Pipeline ===

✓ Data loaded: 4338 customers
✓ Model fitted successfully

=== Cluster Statistics ===
Cluster 0: 3060 customers (70.5%)
Cluster 1: 1061 customers (24.5%)
Cluster 2: 13 customers (0.3%)
Cluster 3: 204 customers (4.7%)

Silhouette score (sample): 0.579
Within-cluster sum of squares: 4092.34
```

This generates two visualization files:
- `cluster_plot.png`: Scatter plot of Frequency vs Monetary, colored by cluster
- `cluster_plot_sizes.png`: Bar chart of cluster sizes

### Prediction Mode

Classify a new customer based on their RFM values:

```bash
cargo run --release -- --input data.csv --clusters 4 --predict "30,10,500.0"
```

**Input format**: `"Recency,Frequency,Monetary"`
- Recency: Days since last purchase (e.g., 30)
- Frequency: Number of transactions (e.g., 10)
- Monetary: Total spending (e.g., 500.0)

**Output:**
```
=== Prediction Mode ===
Input RFM values: R=30, F=10, M=500

✓ Predicted Cluster: 2
  Processing time: 0.39s

Cluster 2 details:
  Size: 3158 customers (72.8% of total)
  Centroid (normalized): R=-0.50, F=-0.02, M=-0.05
```

### Command-Line Options

```
Options:
  -i, --input <INPUT>           Path to CSV file [default: data.csv]
  -k, --clusters <CLUSTERS>     Number of clusters [default: 4]
  -o, --output <OUTPUT>         Output visualization path [default: cluster_plot.png]
  -p, --predict <PREDICT>       Prediction mode: "R,F,M" values
      --max-iters <MAX_ITERS>   K-Means max iterations [default: 300]
      --tolerance <TOLERANCE>   Convergence tolerance [default: 1e-4]
  -v, --verbose                 Enable verbose output
  -h, --help                    Print help
  -V, --version                 Print version
```

## Dataset Requirements

### Expected CSV Format

The application expects a CSV file with the following columns:

| Column       | Type    | Description                          | Example                  |
|-------------|---------|--------------------------------------|--------------------------|
| InvoiceNo   | String  | Transaction identifier               | 536365 or C536365        |
| StockCode   | String  | Product code                         | 85123A                   |
| Description | String  | Product description                  | WHITE HANGING HEART      |
| Quantity    | Integer | Units purchased (negative for returns)| 6                       |
| InvoiceDate | String  | Transaction timestamp                | 12/01/2010 08:26:00      |
| UnitPrice   | Float   | Price per unit                       | 2.55                     |
| CustomerID  | Integer | Unique customer identifier           | 17850                    |
| Country     | String  | Customer country                     | United Kingdom           |

### Date Format

The application supports the format: `MM/DD/YYYY HH:MM:SS`

### Data Cleaning

The pipeline automatically:
- Filters out cancellations (InvoiceNo starting with 'C')
- Removes negative quantities
- Excludes transactions with null CustomerID
- Drops zero or negative prices
- Handles schema inference for mixed-type columns

## Understanding the Results

### Cluster Interpretation

The normalized centroids indicate segment characteristics:

```
Cluster | Recency | Frequency | Monetary
--------|---------|-----------|----------
      0 |   -0.49 |     -0.08 |    -0.08  → Recent, Average frequency/spending
      1 |    1.56 |     -0.35 |    -0.18  → Churned, Low engagement
      2 |   -0.85 |     10.17 |    13.94  → VIP: Recent, High frequency/spending
      3 |   -0.77 |      2.35 |     1.19  → Engaged: Recent, High frequency
```

**Interpretation**:
- **Negative Recency**: Recent purchases (low days since last order)
- **Positive Frequency**: Above-average transaction count
- **Positive Monetary**: Above-average spending

### Quality Metrics

- **Silhouette Score** (0-1): Measures cluster separation. Higher is better. 0.5+ indicates good clustering.
- **Inertia** (Within-cluster sum of squares): Lower values indicate tighter clusters.

### Visualization

The scatter plot uses:
- **X-axis**: Frequency (normalized)
- **Y-axis**: Monetary (normalized)
- **Colors**: RED (Cluster 0), BLUE (Cluster 1), GREEN (Cluster 2), YELLOW (Cluster 3), MAGENTA (Cluster 4)
- **Squares**: Cluster centroids
- **Circles**: Individual customers

## Dependencies

| Crate                | Version | Purpose                                    |
|---------------------|---------|---------------------------------------------|
| polars              | 0.33    | High-performance DataFrame operations       |
| linfa               | 0.7     | Machine learning framework                  |
| linfa-clustering    | 0.7     | K-Means implementation                      |
| linfa-nn            | 0.7     | Distance metrics (L2/Euclidean)            |
| ndarray             | 0.15    | N-dimensional array operations              |
| clap                | 4.0     | Command-line argument parsing               |
| plotters            | 0.3     | Chart and plot generation                   |
| chrono              | 0.4     | Date and time handling                      |
| anyhow              | 1.0     | Error handling and propagation              |
| rand                | 0.8     | Random number generation for K-Means init   |

## Testing

Run the test suite:

```bash
cargo test
```

Tests include:
- Data loading and RFM computation with mock CSV files
- StandardScaler fit/transform correctness
- K-Means model fitting and prediction
- Cluster assignment validation
- Visualization generation (file creation checks)

## Performance Benchmarks

Tested on a dataset with 541,910 transactions:

| Operation              | Time      | Throughput          |
|-----------------------|-----------|---------------------|
| Data loading          | ~150ms    | 3.6M rows/sec       |
| RFM computation       | ~80ms     | 6.8M rows/sec       |
| K-Means (4 clusters)  | ~120ms    | 36k customers/sec   |
| Visualization         | ~20ms     | -                   |
| **Total Pipeline**    | **0.37s** | **1.5M rows/sec**   |

## Use Cases

### Marketing Segmentation

Identify customer segments for targeted campaigns:
- **VIP Customers** (Cluster 2): Recent, frequent, high-spending
- **Loyal Customers** (Cluster 3): Regular purchasers with good spending
- **At-Risk** (Cluster 1): Haven't purchased recently, need re-engagement
- **Casual Buyers** (Cluster 0): Occasional purchasers with average behavior

### Predictive Classification

Use prediction mode to:
- Classify new customers immediately after their first few transactions
- Route customers to appropriate service tiers
- Trigger automated marketing workflows based on segment

### Cohort Analysis

Run clustering at different time periods to track customer journey:
```bash
# Q1 2011
cargo run --release -- --input q1_data.csv --clusters 4 --output q1_clusters.png

# Q2 2011
cargo run --release -- --input q2_data.csv --clusters 4 --output q2_clusters.png
```

## Limitations and Considerations

1. **Cluster Count**: Restricted to 3-5 clusters for interpretability. Adjust in code for other use cases.
2. **Feature Selection**: Only uses RFM. Additional features (product categories, geography) require code modifications.
3. **Distance Metric**: Uses Euclidean distance (L2). Manhattan distance (L1) may be more appropriate for some domains.
4. **Initialization**: Random initialization can lead to different results. Run multiple times or implement K-Means++ for deterministic seeding.
5. **Scalability**: Entire dataset loaded into memory. For >10M customers, consider sampling or streaming approaches.

## Future Enhancements

- **Elbow Method**: Automatic optimal cluster count selection
- **Silhouette Analysis**: Per-cluster quality metrics
- **DBSCAN Support**: Density-based clustering for non-spherical segments
- **Time-Series Analysis**: Tracking segment transitions over time
- **Feature Engineering**: Product category preferences, seasonal patterns
- **Export Formats**: JSON, CSV output for downstream systems
- **Interactive Dashboard**: Web-based visualization and exploration


## Acknowledgments

Built with the excellent Rust data science ecosystem:
- Polars for blazingly fast DataFrame operations
- Linfa for machine learning abstractions
- Plotters for publication-quality visualizations
