import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'french'
filtered_df = df[df['language'] != 'french']
# Convert 'points' to numeric, coercing errors to NaN
filtered_df['points'] = pd.to_numeric(filtered_df['points'], errors='coerce')
# Drop any rows with NaN in 'points' after conversion
filtered_df.dropna(subset=['points'], inplace=True)
# Calculate median of the points
median_points = filtered_df['points'].median()
print(f"Final Answer: {median_points}")