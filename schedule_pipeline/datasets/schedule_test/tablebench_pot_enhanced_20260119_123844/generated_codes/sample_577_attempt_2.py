import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2004
df_filtered = df[df['year'].isin(['2000', '2001', '2002', '2003', '2004'])]
# Convert 'mintage (proof)' to numeric, coercing errors to NaN
df_filtered['mintage (proof)'] = pd.to_numeric(df_filtered['mintage (proof)'], errors='coerce')
# Calculate average mintage (proof)
avg_mintage = df_filtered['mintage (proof)'].mean()
print(f"Final Answer: {avg_mintage:.1f}")