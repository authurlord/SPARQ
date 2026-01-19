import pandas as pd

df = pd.read_csv('table.csv')
# Filter mountains in Greece (case-insensitive, using substring match)
greece_peaks = df[df['country'].str.contains('greece', case=False, na=False)]
# Extract elevation values and compute median
median_elevation = greece_peaks['elevation (m)'].median()
print(f"Final Answer: {median_elevation}")