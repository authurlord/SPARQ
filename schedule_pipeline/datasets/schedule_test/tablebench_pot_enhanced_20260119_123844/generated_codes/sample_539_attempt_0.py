import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'natural growth' to float, handling potential string values
df['natural growth'] = pd.to_numeric(df['natural growth'], errors='coerce')

# Filter data for years 1990 to 2000
df_1990_2000 = df[(df['year'].astype(int) >= 1990) & (df['year'].astype(int) <= 2000)]

# Calculate average natural growth rate
avg_natural_growth = df_1990_2000['natural growth'].mean()

# Define a threshold for "significantly different" (e.g., > 0.5 from average)
threshold = 0.5
significant_years = df_1990_2000[
    (df_1990_2000['natural growth'] - avg_natural_growth).abs() > threshold
]['year'].tolist()

print(f"Final Answer: {', '.join(significant_years)}")