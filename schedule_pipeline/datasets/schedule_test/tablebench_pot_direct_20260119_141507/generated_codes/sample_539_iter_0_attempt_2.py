import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter years from 1990 to 2000 (excluding the summary row)
years_1990_to_2000 = df[df['year'].str.contains(r'199[0-9]|200[0-9]') & (df['year'] != '1990 - 2009')]

# Convert 'natural growth' to float and compute average
natural_growth_values = years_1990_to_2000['natural growth'].astype(float)
average_natural_growth = natural_growth_values.mean()

# Calculate absolute differences from average
differences = abs(natural_growth_values - average_natural_growth)

# Find years where difference is greater than 0.5 (considered "significant")
significant_years = years_1990_to_2000[diffrences > 0.5]['year'].tolist()

# If no such years, return None; otherwise, print the result
if significant_years:
    print(f"Final Answer: {', '.join(significant_years)}")
else:
    print("Final Answer: none")