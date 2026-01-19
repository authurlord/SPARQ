import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'average relative annual growth (%)' to float, handling any non-numeric entries
df['average relative annual growth (%)'] = pd.to_numeric(df['average relative annual growth (%)'], errors='coerce')

# Exclude the total row and rows with NaN values
df_filtered = df[df['country (or dependent territory)'] != 'total'].copy()

# Calculate average growth
avg_growth = 2.42

# Identify countries with absolute deviation > 3%
deviations = df_filtered[
    abs(df_filtered['average relative annual growth (%)'] - avg_growth) > 3
]

# Extract country names
outlier_countries = deviations['country (or dependent territory)'].tolist()

print(f"Final Answer: {', '.join(outlier_countries)}")