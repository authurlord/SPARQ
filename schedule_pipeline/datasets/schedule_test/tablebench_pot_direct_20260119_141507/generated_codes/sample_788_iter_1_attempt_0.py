import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation column to numeric, handling any parsing errors
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')

# Filter rows where country is in Greece (case-insensitive match)
greek_mountains = df[df['country'].str.contains('greece', case=False, na=False)]

# Compute median elevation of Greek mountains
median_elevation = greek_mountains['elevation (m)'].median()
print(f"Final Answer: {median_elevation:.0f}")