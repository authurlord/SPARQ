import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2000 to 2004
filtered_df = df[df['year'].astype(int).between(2000, 2004)]
# Convert 'mintage (proof)' to numeric, treating 'n / a' as NaN
filtered_df['mintage (proof)'] = pd.to_numeric(filtered_df['mintage (proof)'], errors='coerce')
# Calculate average mintage (proof)
average_mintage = filtered_df['mintage (proof)'].mean()
print(f"Final Answer: {average_mintage:.1f}")