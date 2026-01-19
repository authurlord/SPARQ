import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in Greece (including regions like Crete, Euboea, Kefalonia, Samothrace)
greece_peaks = df[df['country'].str.contains('greece', case=False, na=False)]
# Convert elevation to numeric
greece_peaks['elevation (m)'] = pd.to_numeric(greece_peaks['elevation (m)'])
# Calculate median elevation
median_elevation = greece_peaks['elevation (m)'].median()
print(f"Final Answer: {median_elevation}")