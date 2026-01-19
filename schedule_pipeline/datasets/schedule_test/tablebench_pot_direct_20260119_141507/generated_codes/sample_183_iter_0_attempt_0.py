import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns
columns_to_check = ['passengers flown', 'employees (average / year)', 'basic eps']
target_column = 'net profit / loss (sek)'

# Convert string values like '- 2218000000' to numeric (remove space and convert)
df['net profit / loss (sek)'] = df['net profit / loss (sek)'].str.replace('-', '', regex=False).astype(float)
df['passengers flown'] = df['passengers flown'].astype(int)
df['employees (average / year)'] = df['employees (average / year)'].astype(int)
df['basic eps (sek)'] = df['basic eps (sek)'].astype(float)

# Calculate correlation with net profit / loss
correlations = df[columns_to_check].corrwith(df[target_column])

# Find the factor with the highest absolute correlation
significant_factor = correlations.abs().idxmax()

# Check if the correlation is strong enough (e.g., > 0.3)
if correlations.abs().max() < 0.3:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {significant_factor}")