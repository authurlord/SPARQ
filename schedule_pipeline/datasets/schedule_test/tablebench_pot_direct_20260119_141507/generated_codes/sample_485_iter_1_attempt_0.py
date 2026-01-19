import pandas as pd

df = pd.read_csv('table.csv')

# Clean the data: extract 'Top speed' in mph (first column after 'Top speed_1')
df['Top speed (mph)'] = df['Top speed'].str.extract(r'(\d+)$')[0].astype(float)
df['Top speed (mph)'] = df['Top speed (mph)'].fillna(0)

# Convert 'Number' to numeric
df['Number'] = pd.to_numeric(df['Number'], errors='coerce')

# Remove rows with missing 'Number' or 'Top speed'
df = df.dropna(subset=['Top speed (mph)', 'Number'])

# Define outlier detection using IQR
def detect_outliers(data, column_name, factor=1.5):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    return outliers.index.tolist()

# Find outliers in top speed and number
outlier_speeds = detect_outliers(df, 'Top speed (mph)')
outlier_numbers = detect_outliers(df, 'Number')

# Get class names (first column) for those with outliers
classes_with_speed_outliers = df.loc[outlier_speeds, 'Class'].tolist()
classes_with_number_outliers = df.loc[outlier_numbers, 'Class'].tolist()

# Combine unique classes that have deviation
deviant_classes = list(set(classes_with_speed_outliers + classes_with_number_outliers))

# Print result
print(f"Final Answer: {', '.join(deviant_classes)}")