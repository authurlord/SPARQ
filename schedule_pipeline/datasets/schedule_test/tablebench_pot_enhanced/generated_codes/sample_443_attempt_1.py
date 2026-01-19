import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Drop rows with NaN values in critical columns
df_clean = df.dropna(subset=['area (km 2 )', 'population'])

# Function to detect outliers using IQR
def is_outlier(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Detect outliers in area and population
area_outliers = is_outlier(df_clean['area (km 2 )'])
pop_outliers = is_outlier(df_clean['population'])

# Combine outlier flags
outlier_mask = area_outliers | pop_outliers

# Get the names of locations with outliers
outlier_locations = df_clean[outlier_mask]['place'].tolist()

print(f"Final Answer: {', '.join(outlier_locations)}")