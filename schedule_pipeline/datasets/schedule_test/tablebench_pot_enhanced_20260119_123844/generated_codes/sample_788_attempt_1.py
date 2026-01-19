import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in Greece (case-insensitive)
greek_peaks = df[df['country'].str.contains('greece', case=False, na=False)]
# Convert elevation to numeric and compute median
median_elevation = greek_peaks['elevation (m)'].astype(int).median()
print(f"Final Answer: {median_elevation}")