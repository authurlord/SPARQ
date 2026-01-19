import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert registration columns to numeric (in case of string numbers)
for col in ['2002 registrations', '2005 registrations', '2011 registrations']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Identify outliers per year using 2 standard deviations from mean
outliers = []

for year_col in ['2002 registrations', '2005 registrations', '2011 registrations']:
    year_data = df[[year_col, 'breed']].dropna()
    if year_data.empty:
        continue
    
    mean_val = year_data[year_col].mean()
    std_val = year_data[year_col].std()
    if std_val == 0:
        continue
    
    # Find outliers: more than 2 std away from mean
    lower_bound = mean_val - 2 * std_val
    upper_bound = mean_val + 2 * std_val
    
    outliers_in_year = year_data[
        (year_data[year_col] < lower_bound) | (year_data[year_col] > upper_bound)
    ]
    
    if not outliers_in_year.empty:
        for _, row in outliers_in_year.iterrows():
            outliers.append({
                'year': year_col.split()[0],
                'breed': row['breed'],
                'registration': row[year_col]
            })

# Extract unique breed-year combinations
outlier_breeds = set()
for o in outliers:
    outlier_breeds.add(o['breed'])

# Final answer: list of breeds with outliers (as requested)
final_breeds = list(outlier_breeds)

print(f"Final Answer: {', '.join(final_breeds)}")