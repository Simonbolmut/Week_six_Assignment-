import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from seaborn
try:
    iris = sns.load_dataset('iris')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    iris = None

# Display the first few rows
head = iris.head() if iris is not None else None

# Check data types and missing values
info = iris.info() if iris is not None else None
missing_values = iris.isnull().sum() if iris is not None else None

# Clean data by dropping missing values (if any)
iris_clean = iris.dropna() if iris is not None else None

# Basic statistics
describe = iris_clean.describe() if iris_clean is not None else None

# Grouping by species and calculating the mean
grouped_mean = iris_clean.groupby('species').mean(numeric_only=True) if iris_clean is not None else None

# Create visualizations
if iris_clean is not None:
    # Set plot style
    sns.set(style="whitegrid")

    # Line chart (synthetic example - using index as time)
    plt.figure(figsize=(8, 5))
    plt.plot(iris_clean.index, iris_clean['sepal_length'], label='Sepal Length')
    plt.title('Trend of Sepal Length Over Index')
    plt.xlabel('Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x='species', y='petal_length', data=iris_clean, estimator='mean', ci=None)
    plt.title('Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Average Petal Length (cm)')
    plt.tight_layout()
    plt.show()

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(iris_clean['sepal_width'], bins=10, edgecolor='black')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris_clean)
    plt.title('Sepal Length vs. Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

(head, missing_values, describe, grouped_mean)
