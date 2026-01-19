import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for years 1990 to 2000
df_1990_2000 = df[(df['year'] >= '1990') & (df['year'] <= '2000')]

# Convert 'natural growth' to float for calculation
df_1990_2000['natural growth'] = pd.to_numeric(df_1990_2000['natural growth'])

# Calculate average natural growth rate from 1990 to 2000
avg_natural_growth = df_1990_2000['natural growth'].mean()

# Find years where natural growth is significantly different (more than 0.5 away from average)
threshold = 0.5
diff = abs(df_1990_2000['natural growth'] - avg_natural_growth)
significant_years = df_1990_2000[diff > threshold]['year']

print(f"Final Answer: {', '.join(significant_years)}")