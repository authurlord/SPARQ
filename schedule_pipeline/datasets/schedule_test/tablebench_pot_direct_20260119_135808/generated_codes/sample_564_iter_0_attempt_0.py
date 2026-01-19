import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2004 to 2013
df_filtered = df[(df['Year'] >= '2004') & (df['Year'] <= '2013')]
# Convert 'No. of performances' to numeric, coercing errors to NaN
df_filtered['No. of performances'] = pd.to_numeric(df_filtered['No. of performances'], errors='coerce')
# Calculate average performances
avg_performances = df_filtered['No. of performances'].mean()
print(f"Final Answer: {avg_performances:.1f}")