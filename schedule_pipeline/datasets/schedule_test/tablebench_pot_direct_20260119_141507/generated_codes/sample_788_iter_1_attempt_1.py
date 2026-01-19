import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation to numeric, handling any parsing errors
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')

# Filter mountains in Greece (case-insensitive match)
greece_peaks = df[df['country'].str.contains('greece', case=False, na=False)]

# Calculate median elevation of peaks in Greece
median_elevation = greece_peaks['elevation (m)'].median()
print(f"Final Answer: {median_elevation:.0f}")