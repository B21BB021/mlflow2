# -*- coding: utf-8 -*-


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

"""##2. Load and Explore the Dataset




"""

# Load the Titanic dataset directly from seaborn
data = sns.load_dataset('titanic')
# Display initial dataset information
print("Initial Dataset Info:\n")
print(data.info())

# Display the first few rows of the dataset
print("\nFirst Few Rows of the Dataset:\n")
print(data.head())

# Display the last few rows of the dataset
print("\nLast Few Rows of the Dataset:\n")
print(data.tail())

# Display the shape of the dataset (number of rows and columns)
print("\nShape of the Dataset (rows, columns):\n")
print(data.shape)

"""## 3.Check for Missing values"""

# Calculate the count of missing values in each column
missing_values = data.isnull().sum()

# Calculate the percentage of missing values in each column
missing_percentage = (missing_values / len(data)) * 100

# Display missing values count and percentage
print("\nMissing Values Count:\n", missing_values)
print("\nMissing Values Percentage:\n", missing_percentage)

"""## 4. Handle Missing Values"""

# Replacing missing values in 'Age' with the mean age
data['age'].fillna(data['age'].mean(), inplace=True)

# Replacing missing values in 'Embarked' with the mode (most frequent value)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

# Replacing missing values in 'Fare' with zero
data['fare'].fillna(0, inplace=True)

# Adding 'Unknown' as a new category in 'Deck'
data['deck'] = data['deck'].cat.add_categories('Unknown')

# Replacing missing values in 'Deck' with 'Unknown'
data['deck'].fillna('Unknown', inplace=True)

# Displaying the updated dataset
print("\nMissing Values Count After Handling 'Deck':\n", data.isnull().sum())

# Replacing missing values in 'Embark_town' with the mode (most frequent value)
data['embark_town'].fillna(data['embark_town'].mode()[0], inplace=True)

# Display missing values count after handling
print("\nMissing Values Count After Handling:\n", data.isnull().sum())

# Calculate the percentage of missing values after imputation
missing_percentage_after = (data.isnull().sum() / len(data)) * 100

# Display missing values percentage after handling
print("\nMissing Values Percentage After Handling:\n", missing_percentage_after)

"""##5. Remove Duplicates"""

# Write code for - Removing duplicates ensures that each entry in the dataset is unique.

print(f"\nDuplicates removed. Number of rows now: {len(data)}")

"""##6. Correct Data Types"""

# Convert 'age' and 'fare' to float
data['age'] = data['age'].astype(float)
data['fare'] = data['fare'].astype(float)

# Convert 'sex' and 'embarked' to categorical
data['sex'] = data['sex'].astype('category')
data['embarked'] = data['embarked'].astype('category')

# Display updated data types
print("\nUpdated Data Types:\n")
print(data.dtypes)

"""##7. Normalize or Standardize the Data"""

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the 'age' and 'fare' columns
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])

# Display the first few rows of the dataset to check the changes
print("\nFirst Few Rows After Standardizing 'age' and 'fare':\n")
print(data[['age', 'fare']].head())

"""##8. Create New Features (Feature Engineering)"""

# Create the new feature 'family_size'
data['family_size'] = data['sibsp'] + data['parch'] + 1

# Display the first few rows of the dataset to check the new feature
print("\nFirst Few Rows After Creating 'family_size':\n")
print(data[['sibsp', 'parch', 'family_size']].head())

"""##9. Aggregation"""

# Group by 'pclass' and calculate the mean of 'age' and 'fare'
agg_data = data.groupby('pclass')[['age', 'fare']].mean()

# Display the aggregated data
print("\nAggregated Data by Pclass:\n", agg_data)

"""##10. Outlier Detection and Removal"""

# Plotting boxplot to detect outliers in 'fare'
plt.figure(figsize=(8,6))
sns.boxplot(x=data['fare'])
plt.title("Boxplot of 'Fare' to Detect Outliers")
plt.show()

# Removing outliers where 'Fare' is greater than 200
data = data[data['fare'] < 200]

# Display the message after removing outliers
print("\nOutliers removed based on Fare > 200.\n")

# Display the first few rows of the dataset after outlier removal
print(data[['fare']].head())

"""##11. Separate Numerical and Categorical Variables"""

# Explanation: Separating variables helps in organizing the data for more targeted analysis.

numerical_features = data.select_dtypes(include=['float64', 'int64'])
categorical_features = data.select_dtypes(include=['object', 'category', 'bool'])

print("\nNumerical Features:\n", numerical_features.head())
print("\nCategorical Features:\n", categorical_features.head())

"""##12. Data Visualization"""

# Explanation: Visualization helps to understand the distribution, relationships, and patterns in the data.

# Univariate Analysis: Age and Fare Distribution
plt.figure(figsize=(8,6))
sns.histplot(data['age'], kde=True, bins=20)
plt.title("Standardized Age Distribution")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(data['fare'], kde=True, bins=20)
plt.title("Standardized Fare Distribution")
plt.show()

# Bivariate Analysis: Scatterplot of Age vs Fare
plt.figure(figsize=(8,6))
sns.scatterplot(x=data['age'], y=data['fare'])
plt.title("Age vs Fare")
plt.show()

# Heatmap to show correlation between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(data[['age', 'fare', 'sibsp', 'parch', 'family_size']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap of Correlation Between Features")
plt.show()

# Boxplot of 'Pclass' vs 'Fare'
plt.figure(figsize=(8,6))
sns.boxplot(x='pclass', y='fare', data=data)
plt.title('Boxplot of Pclass vs Fare')
plt.show()

"""##13. Descriptive Statistics"""

# Calculate measures of central tendency for 'age' and 'fare'
mean_age = data['age'].mean()
median_age = data['age'].median()
mode_age = data['age'].mode()[0]

mean_fare = data['fare'].mean()
median_fare = data['fare'].median()
mode_fare = data['fare'].mode()[0]

print(f"\nMean Age: {mean_age}, Median Age: {median_age}, Mode Age: {mode_age}")
print(f"Mean Fare: {mean_fare}, Median Fare: {median_fare}, Mode Fare: {mode_fare}")

# Calculate measures of dispersion for 'age' and 'fare'
range_age = data['age'].max() - data['age'].min()
variance_age = data['age'].var()
std_dev_age = data['age'].std()

range_fare = data['fare'].max() - data['fare'].min()
variance_fare = data['fare'].var()
std_dev_fare = data['fare'].std()

print(f"\nAge - Range: {range_age}, Variance: {variance_age}, Std Dev: {std_dev_age}")
print(f"Fare - Range: {range_fare}, Variance: {variance_fare}, Std Dev: {std_dev_fare}")

# Summary of the data (Descriptive Statistics)
print("\nSummary Statistics:\n", data.describe())

"""#Linear Regression"""

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data here (assuming 'data' is already available in your notebook or you have saved somwhere else)
data = sns.load_dataset('titanic')

"""##2: Check Column Names"""

# Check the column names in the dataset
print("Columns in the dataset:", data.columns)

"""##3: Drop Irrelevant Columns"""

# Drop irrelevant columns
columns_to_drop = ['survived', 'who', 'alive', 'alone']
data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

# Check the updated columns after dropping irrelevant ones
print("Features in the dataset:", data_cleaned.columns)

"""##4: Define Features"""

# Define numerical features
numerical_features = ['age', 'sibsp', 'parch', 'family_size']

# Define categorical features
categorical_features = ['sex', 'embarked', 'class', 'adult_male', 'deck', 'embark_town', 'pclass']

# Check if the columns exist in the DataFrame
print("Numerical Features:", [feature for feature in numerical_features if feature in data_cleaned.columns])
print("Categorical Features:", [feature for feature in categorical_features if feature in data_cleaned.columns])

"""##5: Prepare Features and Target Variable"""

# Define the target variable and features
target_variable = 'fare'
features = [col for col in data_cleaned.columns if col != target_variable]

# Separate features and target variable
X = data_cleaned[features]
y = data_cleaned[target_variable]

# Check the shapes of X and y to ensure separation is correct
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

"""##6: Handle Categorical Variables and Standardize Numerical Features"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define numerical features and categorical features
numerical_features = ['age', 'sibsp', 'parch']  # Adjust as needed based on your dataset
categorical_features = ['sex', 'embarked', 'class', 'adult_male', 'deck', 'embark_town', 'pclass']  # Adjust as needed

# Check available columns in X
print("Available columns in X:", X.columns)

# Check which categorical features are actually present
existing_categorical_features = [feature for feature in categorical_features if feature in X.columns]
print("Existing categorical features:", existing_categorical_features)

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=existing_categorical_features, drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

# Check the resulting dataframe
print(X_encoded.head())
print("Features Shape after Encoding and Scaling:", X_encoded.shape)

"""##7: Split Dataset"""

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting splits
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

"""##8: Initialize and Train the Model"""

from sklearn.impute import SimpleImputer

# Initialize the imputer to replace NaN with the mean value of each column
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both training and testing data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the imputed training data
model.fit(X_train_imputed, y_train)

# Print a message indicating that the model has been trained
print("Linear Regression model has been fitted to the training data.")

"""##9: Predict and Evaluate"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Predict the fare on the test data
y_pred = model.predict(X_test_imputed)  # Make sure to use imputed test data

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")

"""##10: Visualize Results"""

import matplotlib.pyplot as plt

# Visualizing actual vs predicted fares
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue", label="Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Ideal Fit")
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Actual vs Predicted Fare')
plt.legend()
plt.show()
