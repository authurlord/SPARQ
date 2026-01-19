import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in Greece (excluding those with specific regions like Crete, Euboea, etc., but including all Greece entries)
greek_peaks = df[df['country'].str.lower().str.contains('greece', na=False)]
# Convert elevation to numeric
greek_peaks['elevation (m)'] = pd.to_numeric(greek_peaks['elevation (m)'])
# Calculate median elevation
median_elevation = greek_peaks['elevation (m)'].median()
print(f"Final Answer: {median_elevation}")