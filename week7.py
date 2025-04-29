# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris_data = load_iris(as_frame=True)
    iris = iris_data.frame  # Extract dataframe
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("First five rows of the dataset:")
print(iris.head())

# Check the structure (data types and missing values)
print("\nDataset Info:")
print(iris.info())

# Check for missing values
print("\nMissing Values:")
print(iris.isnull().sum())

# Clean the dataset (fill missing values if there were any)
if iris.isnull().values.any():
    iris = iris.fillna(method='ffill')
    print("Missing values filled.\n")
else:
    print("No missing values to clean.\n")

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(iris.describe())

# Perform groupings (mean petal length per target)
iris['target_name'] = iris['target'].map(dict(enumerate(iris_data.target_names)))
grouped = iris.groupby('target_name')['petal length (cm)'].mean()
print("\nAverage petal length per species:")
print(grouped)

# Identify patterns
print("\nFindings:")
print("- Setosa species have notably shorter petal lengths.")
print("- Virginica species have the longest petals on average.")

# Task 3: Data Visualization

# Custom style
sns.set(style="whitegrid")

# Line chart (simulate a time-series by adding a fake 'day' column)
iris['day'] = range(1, len(iris) + 1)
plt.figure(figsize=(10,6))
plt.plot(iris['day'], iris['sepal length (cm)'], label='Sepal Length', color='blue')
plt.plot(iris['day'], iris['petal length (cm)'], label='Petal Length', color='green')
plt.title('Sepal and Petal Length over Days', fontsize=16)
plt.xlabel('Day', fontsize=12)
plt.ylabel('Length (cm)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Bar chart (average petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x=grouped.index, y=grouped.values, palette='coolwarm')
plt.title('Average Petal Length per Species', fontsize=16)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Average Petal Length (cm)', fontsize=12)
plt.show()

# Histogram (distribution of sepal width)
plt.figure(figsize=(8,5))
plt.hist(iris['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width', fontsize=16)
plt.xlabel('Sepal Width (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Scatter plot (sepal length vs. petal length)
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target_name', data=iris, palette='deep')
plt.title('Sepal Length vs Petal Length', fontsize=16)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.legend(title='Species')
plt.show()