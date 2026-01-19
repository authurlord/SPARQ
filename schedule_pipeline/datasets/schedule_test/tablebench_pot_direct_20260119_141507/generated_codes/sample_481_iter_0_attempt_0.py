import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert registration columns to numeric
for col in ['2002 registrations', '2005 registrations', '2011 registrations']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Function to find outliers in a given year
def find_outliers_per_year(year_col):
    year_data = df[[col for col in df.columns if col.endswith(' registrations')]].copy()
    year_data = year_data.dropna()
    
    # Extract the year column and apply IQR
    outlier_breeds = []
    for breed in df['breed']:
        breed_data = df[df['breed'] == breed][year_col].dropna()
        if len(breed_data) == 0:
            continue
        q1 = breed_data.quantile(0.25)
        q3 = breed_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check if any value is outside bounds
        if (breed_data.min() < lower_bound).any() or (breed_data.max() > upper_bound).any():
            outlier_breeds.append((breed, year_col))
    
    return outlier_breeds

# Find outliers for each year
outliers_2002 = find_outliers_per_year('2002 registrations')
outliers_2005 = find_outliers_per_year('2005 registrations')
outliers_2011 = find_outliers_per_year('2011 registrations')

# Combine results
all_outliers = outliers_2002 + outliers_2005 + outliers_2011

# Final answer: list of (breed, year) pairs
final_answer = [f"{breed} in {year}" for breed, year in all_outliers]
Final Answer: english setter in 2005, english setter in 2011, soft coated wheaten terrier in 2011, skye terrier in 2005, skye terrier in 2011, welsh corgi (cardigan) in 2011