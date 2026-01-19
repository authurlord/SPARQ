import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert registration columns to numeric
df['2002 registrations'] = pd.to_numeric(df['2002 registrations'], errors='coerce')
df['2005 registrations'] = pd.to_numeric(df['2005 registrations'], errors='coerce')
df['2011 registrations'] = pd.to_numeric(df['2011 registrations'], errors='coerce')

# Function to compute z-scores and identify outliers
def find_outliers_per_year(year_col):
    # Remove rows with NaN
    valid_data = df.dropna(subset=[year_col])
    mean_val = valid_data[year_col].mean()
    std_val = valid_data[year_col].std()
    if std_val == 0:
        return []
    z_scores = np.abs((valid_data[year_col] - mean_val) / std_val)
    outliers = valid_data[valid_data[year_col].apply(lambda x: abs(z_scores.iloc[valid_data.index.get_loc(x)]) > 3)]
    return list(outliers.index)

# Identify outliers for each year
outlier_breeds = set()
for year in ['2002 registrations', '2005 registrations', '2011 registrations']:
    year_outliers = find_outliers_per_year(year)
    outlier_breeds.update(year_outliers)

# Convert to list and print
final_outliers = list(outlier_breeds)
print(f"Final Answer: {', '.join(final_outliers)}")