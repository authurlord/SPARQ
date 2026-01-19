import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years between 2004 and 2013
df_filtered = df[(df['Year'].astype(int) >= 2004) & (df['Year'].astype(int) <= 2013)]
# Convert 'No. of performances' to numeric, coercing errors to NaN
df_filtered['No. of performances'] = pd.to_numeric(df_filtered['No. of performances'], errors='coerce')
# Calculate average number of performances
avg_performances = df_filtered['No. of performances'].mean()
print(f"Final Answer: {avg_performances:.1f}")