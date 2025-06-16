import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 🧪 Load the dataset
df = pd.read_csv("train.csv")  # Make sure train.csv is in the same folder
print("Data Loaded Successfully.\n")
print("First 5 rows of the dataset:\n", df.head())

# 📋 Basic Information
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 🧹 Data Cleaning
# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column due to many missing values
df.drop(columns=['Cabin'], inplace=True)

# Confirm missing values again
print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# 🔍 Descriptive Statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# 🧠 Exploratory Data Analysis

# 1. Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', palette='Set2')
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sex', hue='Survived', palette='Set1')
plt.title("Survival by Gender")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title='Survived')
plt.tight_layout()
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set3')
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title='Survived')
plt.tight_layout()
plt.show()

# 4. Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Age', bins=30, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 5. Fare Distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Fare', bins=30, kde=True, color='green')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 6. Heatmap of Correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation")
plt.tight_layout()
plt.show()
