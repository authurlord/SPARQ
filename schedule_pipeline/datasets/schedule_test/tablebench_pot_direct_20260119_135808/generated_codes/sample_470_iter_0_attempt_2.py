import pandas as pd

df = pd.read_csv('table.csv')

# Convert the economic columns to numeric
df['2011 (imf)'] = pd.to_numeric(df['2011 (imf)'])
df['2008 (cia factbook)'] = pd.to_numeric(df['2008 (cia factbook)'])

# Function to detect outliers using IQR
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Detect outliers in both columns
outliers_imf = detect_outliers_iqr(df['2011 (imf)'])
outliers_cia = detect_outliers_iqr(df['2008 (cia factbook)'])

# Combine outliers from both columns
outlier_mask = outliers_imf | outliers_cia

# Get the nations with outliers
outlier_nations = df[outlier_mask]['nation'].tolist()

print(f"Final Answer: {', '.join(outlier_nations)}")