import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'natural growth' to float
df['natural growth'] = pd.to_numeric(df['natural growth'], errors='coerce')

# Filter data from 1990 to 2000
df_1990_2000 = df[(df['year'] >= '1990') & (df['year'] <= '2000')]

# Calculate average natural growth rate
avg_growth = df_1990_2000['natural growth'].mean()

# Identify years with significantly different natural growth (e.g., > 1.5x average)
# We'll define "significantly different" as values outside the range [avg - 0.5, avg + 0.5] for simplicity
threshold = 0.5
diff_years = df_1990_2000[abs(df_1990_2000['natural growth'] - avg_growth) > threshold]['year']

# Output the result
print(f"Final Answer: {', '.join(diff_years)}")