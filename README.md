# Electric Vehicle Population Data Analysis

This project provides an **Exploratory Data Analysis (EDA)** tool for analyzing Electric Vehicle (EV) population data using **Python**, **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**. It includes data cleaning, outlier detection, visualizations, and statistical analysis to uncover trends and patterns in the EV market.

## Features

- **Data Loading & Cleaning**: Automatically handles missing values and imputes missing data.
- **Exploratory Data Analysis**: Summary statistics, null value checks, skewness detection.
- **Visualizations**: 
  - Bar plot of top EV manufacturers
  - Histogram of EV model years
  - Correlation heatmap
  - Boxplots, scatter plots, and line plots
- **Outlier Detection**: Identifies outliers using IQR and Z-score methods.
- **Statistical Analysis**: 
  - Correlation and covariance matrix
  - T-test for comparing electric range based on model year

## Getting Started

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ev-data-analysis.git
    cd ev-data-analysis
    ```

2. **Install dependencies**:
    Make sure you have Python 3.x installed. Install the required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the script**:
    Update the `file_path` variable in the `EVDataModel` class to point to your local dataset, then run the script:
    ```bash
    python ev_analysis.py
    ```

## Dataset

This project assumes the presence of a **CSV** dataset of Electric Vehicles with columns like:

- `Make` (EV Manufacturer)
- `Model Year`
- `Electric Range`
- Other EV-related attributes (e.g., Vehicle Type, VIN)

Example dataset: [Washington State EV Registry Dataset](https://catalog.data.gov/dataset/electric-vehicle-population-data)

## Example Outputs

- **Top 10 EV Manufacturers**: Bar plot showing the most common EV makes.
- **EV Model Year Distribution**: Histogram of EV model years.
- **Correlation Heatmap**: Heatmap visualizing relationships between numerical features.
- **Electric Range Outliers**: Boxplots and statistical summaries to detect outliers.

## Dependencies

This project requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

Install them using the following command:

```bash
pip install -r requirements.txt
