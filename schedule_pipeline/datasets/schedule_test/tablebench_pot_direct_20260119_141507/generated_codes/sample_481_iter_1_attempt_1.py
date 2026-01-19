import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert registration columns to numeric
df['2002 registrations'] = pd.to_numeric(df['2002 registrations'], errors='coerce')
df['2005 registrations'] = pd.to_numeric(df['2005 registrations'], errors='coerce')
df['2011 registrations'] = pd.to_numeric(df['2011 registrations'], errors='coerce')

# List to store outlier findings
outliers = []

# Define function to detect outliers in a given year
def detect_outliers(year_col, year_name):
    # Extract the column for this year
    year_data = df[year_col]
    if year_data.isnull().all():
        return []
    
    mean_val = year_data.mean()
    std_val = year_data.std()
    if std_val == 0:
        return []
    
    # Find values more than 2 standard deviations from the mean
    z_scores = np.abs((year_data - mean_val) / std_val)
    outliers_in_year = year_data[z_scores > 2].index
    outlier_breeds = df.loc[outliers_in_year, 'breed'].tolist()
    return [(breed, year_name) for breed in outlier_breeds]

# Detect outliers for each year
outlier_results = []
outlier_results.extend(detect_outliers('2002 registrations', '2002'))
outlier_results.extend(detect_outliers('2005 registrations', '2005'))
outlier_results.extend(detect_outliers('2011 registrations', '2011'))

# Final answer: list of (breed, year) pairs with outliers
print(f"Final Answer: {', '.join([f'{breed} in {year}' for breed, year in outlier_results])}")