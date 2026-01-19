import pandas as pd

df = pd.read_csv('table.csv')
# Filter mountains in Greece (case-insensitive match)
greek_mountains = df[df['country'].str.contains('greece', case=False, na=False)]
# Extract elevation values and compute median
median_elevation = greek_mountains['elevation (m)'].median()
print(f"Final Answer: {median_elevation}")