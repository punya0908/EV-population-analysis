# Electric Vehicle Population Data

# Objectives:
# 1 Load and Manipulate Data Using Pandas and NumPy
# 2 Perform Exploratory Data Analysis to Uncover Patterns
# 3 Analyze Feature Relationships Using Correlation and Covariance
# 4 Detect and Interpret Outliers in the Dataset
# 5 Visualize the data using graphs and charts for insights
# (Bar Plot, Histogram, Heatmap, Box Plot, etc.)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

class EVDataModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        print("✅ Data loaded.")

    def head(self): # few data items can be added here
        print("📊 First 5 rows:")
        print(self.df.head())

    def info(self):
        '''
        print("📋 Data Types:")
        print(self.df.dtypes)
        print("\n❓ Null Values:")
        print(self.df.isnull().sum())
        '''

        # Data Type Overview
        print("📋 **Data Types Overview**:")
        print(self.df.dtypes)
        
        # Null Value Summary
        print("\n❓ **Missing Data Summary**:")
        missing_data = self.df.isnull().sum()
        missing_data_percent = (missing_data / len(self.df)) * 100
        missing_info = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage': missing_data_percent.round(2)
        })
        print(missing_info[missing_info['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False))
        
        # Total Records and Columns in the Dataset
        print("\n📝 **Data Shape:**")
        print(f"Total Records: {self.df.shape[0]}")
        print(f"Total Columns: {self.df.shape[1]}")

    def describe(self): # or we can add here also
        print("📈 Summary Statistics:")
        print(self.df.describe())

    def correlation_matrix(self):
        print("🔗 Correlation Matrix:")
        print(self.df.corr(numeric_only=True))

    def covariance_matrix(self):
        print("🔀 Covariance Matrix:")
        print(self.df.cov(numeric_only=True))

    def barplot_top_makes(self):
        top_makes = self.df['Make'].value_counts().head(10)
        sns.barplot(x=top_makes.values, y=top_makes.index, palette='crest')
        plt.title("Top 10 EV Makes")
        plt.xlabel("Count")
        plt.ylabel("Make")
        plt.tight_layout()
        plt.show()

    def histogram_model_year(self):
        sns.histplot(self.df['Model Year'], bins=20, kde=True, color='orchid')
        plt.title("Distribution of EV Model Years")
        plt.xlabel("Model Year")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def heatmap_corr(self):
        '''
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='Spectral')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        '''

        corr = self.df.corr(numeric_only=True)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap of Numerical Features", fontsize=16, weight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        plt.show()

    def boxplot_column(self, col):
        ''' # this is if you wanna see all numerical columns in different plots
        num_cols = self.df.select_dtypes(include=['number']).columns
        sns.boxplot(data=self.df[num_cols], color='red')
        for col in num_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df[col], color='red')
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()
        '''

        sns.boxplot(x=self.df[col], color='green')
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

    def detect_outliers_iqr(self, col):
        q1 = self.df[col].quantile(0.25)
        q3 = self.df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
        print(f"🚨 IQR Outliers in '{col}':", len(outliers))
        self.boxplot_column(col)

    def detect_outliers_zscore(self, col):
        z_scores = zscore(self.df[col])
        outliers = self.df[abs(z_scores) > 3]
        print(f"🚨 Z-Score Outliers in '{col}':", len(outliers))
        self.boxplot_column(col)
    
    def handle_missing_values(self):
        num_cols = self.df.select_dtypes(include=['number']).columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        for col in cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        print("✅ Missing values handled: Numerical → Mean, Categorical → Mode.")

    def line_plot(self, x_col, y_col):
        # plt.plot(self.df[x_col], self.df[y_col], color='blue')
        # plt.title(f"Line Plot: {y_col} over {x_col}")
        # plt.xlabel(x_col)
        # plt.ylabel(y_col)
        # plt.tight_layout()
        # plt.show()
        grouped = self.df.groupby("Model Year")["Electric Range"].mean()
        grouped.plot(marker='o', color='blue', linestyle='-')
        plt.title("Average Electric Range by Model Year")
        plt.xlabel("Model Year")
        plt.ylabel("Average Electric Range")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def scatter_plot(self, x_col, y_col):
        sns.scatterplot(data=self.df, x=x_col, y=y_col, color='green')
        plt.title(f"Scatter Plot: {y_col} vs {x_col}")
        plt.tight_layout()
        plt.show()

    def check_skewness(self):
        num_cols = self.df.select_dtypes(include=['number']).columns
        print("📐 Skewness for Numerical Columns:")
        for col in num_cols:
            print(f"{col}: {self.df[col].skew().round(2)}")

    def hist_all_numerical(self):
        num_cols = self.df.select_dtypes(include=['number']).columns
        self.df[num_cols].hist(bins=20, figsize=(12, 8), color='skyblue')
        plt.suptitle("Histograms for All Numerical Columns", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()

    def t_test_groups(self, col, group_col, split_value):
        from scipy.stats import ttest_ind
        group1 = self.df[self.df[group_col] <= split_value][col].dropna()
        group2 = self.df[self.df[group_col] > split_value][col].dropna()
        stat, p_val = ttest_ind(group1, group2)
        print(f"🧪 T-Test on '{col}' grouped by '{group_col}' <= {split_value}")
        print(f"T-Statistic: {stat}")
        print(f"P-Value: {p_val}")
        if p_val < 0.05:
            print("✅ Statistically Significant Difference")
        else:
            print("❌ No Statistically Significant Difference")

# --------------------------code calling is here-------------------------------------------

if __name__ == "__main__":
    model = EVDataModel(r"D:\VS code\python_lpu\EV-data\Electric_Vehicle_Population_Data.csv")

    print("\n📌 Menu:")
    print("1 - Show first rows")
    print("2 - Show data types & nulls")
    print("3 - Summary statistics")
    print("4 - Correlation matrix")
    print("5 - Covariance matrix")
    print("6 - Barplot (Top EV Makes)")
    print("7 - Histogram (Model Year)")
    print("8 - Heatmap (Correlation)")
    print("9 - Boxplot (Electric Range)") # this can be edited in multiple ways
    print("10 - IQR Outliers (Electric Range)")
    print("11 - Z-Score Outliers (Electric Range)")
    print("12 - Handle Missing Values")
    print("13 - Line Plot (Model Year vs Electric Range)") # i have to check this one, THIS IS NOT TO BE ADDED
    print("14 - Scatter Plot (Model Year vs Electric Range)")
    print("15 - Skewness of Numerical Columns")
    print("16 - Histograms (All Numerical in One Page)")
    print("17 - T-Test (Electric Range based on Model Year)")

    print("0 - Exit")

    choice = input("\nEnter option: ")

    actions = {
        "1": model.head,
        "2": model.info,
        "3": model.describe,
        "4": model.correlation_matrix,
        "5": model.covariance_matrix,
        "6": model.barplot_top_makes,
        "7": model.histogram_model_year,
        "8": model.heatmap_corr,
        "9": lambda: model.boxplot_column("Electric Range"),
        "10": lambda: model.detect_outliers_iqr("Electric Range"),
        "11": lambda: model.detect_outliers_zscore("Electric Range"),
        "12": model.handle_missing_values,
        "13": lambda: model.line_plot("Model Year", "Electric Range"),
        "14": lambda: model.scatter_plot("Model Year", "Electric Range"),
        "15": model.check_skewness,
        "16": model.hist_all_numerical,
        "17": lambda: model.t_test_groups("Electric Range", "Model Year", 2020)
    }

    while choice != "0":
        func = actions.get(choice)
        if func:
            func()
        else:
            print("❌ Invalid choice.")
        choice = input("\nEnter option: ")

    print("👋 Exiting.")
