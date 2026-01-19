import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for years 1990 to 2000
df_1990_2000 = df[(df['year'] >= '1990') & (df['year'] <= '2000')]

# Convert 'natural growth' to float
df_1990_2000['natural growth'] = pd.to_numeric(df_1990_2000['natural growth'])

# Calculate average natural growth rate from 1990 to 2000
avg_natural_growth = df_1990_2000['natural growth'].mean()

# Identify years with significant deviation (>0.5 from average)
significant_years = df_1990_2000[abs(df_1990_2000['natural growth'] - avg_natural_growth) > 0.5]['year']

# Print the result
print(f"Final Answer: {', '.join(significant_years)}")