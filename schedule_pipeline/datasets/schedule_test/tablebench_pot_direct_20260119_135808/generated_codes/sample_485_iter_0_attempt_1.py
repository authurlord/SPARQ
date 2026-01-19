import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Top speed' and 'Number' to numeric, handling non-numeric entries
df['Top speed'] = pd.to_numeric(df['Top speed'], errors='coerce')
df['Number'] = pd.to_numeric(df['Number'], errors='coerce')

# Drop rows with missing values in these columns
df_clean = df.dropna(subset=['Top speed', 'Number'])

# Function to detect outliers using IQR
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

# Detect outliers in 'Top speed' and 'Number'
outliers_speed = detect_outliers(df_clean['Top speed'])
outliers_number = detect_outliers(df_clean['Number'])

# Get the class names of the outliers
outlier_classes_speed = df_clean.loc[df_clean['Top speed'].isin(outliers_speed), 'Class']
outlier_classes_number = df_clean.loc[df_clean['Number'].isin(outliers_number), 'Class']

# Combine and get unique outlier classes
outlier_classes = set(outlier_classes_speed).union(set(outlier_classes_number))

print(f"Final Answer: {', '.join(outlier_classes)}")