import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to numeric for filtering, and 'natural growth' to float
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['natural growth'] = pd.to_numeric(df['natural growth'], errors='coerce')

# Filter data for 1990 to 2000
df_1990_2000 = df[(df['year'] >= 1990) & (df['year'] <= 2000)]

# Calculate average natural growth rate
avg_natural_growth = df_1990_2000['natural growth'].mean()

# Define significant difference threshold (e.g., > 0.5 from average)
threshold = 0.5
significant_years = df_1990_2000[abs(df_1990_2000['natural growth'] - avg_natural_growth) > threshold]['year']

# Convert to list of integers and print
print(f"Final Answer: {int(significant_years.iloc[0])}")