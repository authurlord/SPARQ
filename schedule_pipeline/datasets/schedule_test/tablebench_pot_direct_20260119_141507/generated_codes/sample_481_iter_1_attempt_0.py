import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert registration columns to numeric (in case of strings)
for col in ['2002 registrations', '2005 registrations', '2011 registrations']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define function to find breeds with max/min in a given year
def find_extreme_breeds(year_col):
    year_data = df[[col for col in df.columns if col.startswith('200')]][year_col]
    max_val = year_data.max()
    min_val = year_data.min()
    max_breed = df[df[year_col] == max_val]['breed'].values[0]
    min_breed = df[df[year_col] == min_val]['breed'].values[0]
    return max_breed, min_breed

# Extract years
years = ['2002 registrations', '2005 registrations', '2011 registrations']

# Find extreme breeds per year
results = {}
for year in years:
    max_breed, min_breed = find_extreme_breeds(year)
    results[year] = (max_breed, min_breed)

# Output the results
print(f"Final Answer: {results['2002 registrations'][0]}, {results['2002 registrations'][1]}, {results['2005 registrations'][0]}, {results['2005 registrations'][1]}, {results['2011 registrations'][0]}, {results['2011 registrations'][1]}")