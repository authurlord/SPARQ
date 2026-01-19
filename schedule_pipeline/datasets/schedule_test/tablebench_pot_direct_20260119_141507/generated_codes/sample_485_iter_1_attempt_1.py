import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean and convert relevant columns
df['Top speed'] = df['Top speed'].str.replace('mph', '').str.replace('km/h', '').astype(float)
df['Number'] = pd.to_numeric(df['Number'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['Top speed', 'Number'])

# Define outlier detection using IQR method
def detect_outliers(df, column, k=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in top speed and number
outlier_speed = detect_outliers(df, 'Top speed')
outlier_number = detect_outliers(df, 'Number')

# Extract class names that have outliers
outlier_classes_speed = outlier_speed['Class'].tolist()
outlier_classes_number = outlier_number['Class'].tolist()

# Combine unique classes with deviations
deviant_classes = list(set(outlier_classes_speed + outlier_classes_number))

# Final answer: list of classes that deviate significantly
print(f"Final Answer: {', '.join(deviant_classes)}")