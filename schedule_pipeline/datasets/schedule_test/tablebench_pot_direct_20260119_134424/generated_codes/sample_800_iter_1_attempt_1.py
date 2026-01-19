import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'french'
filtered_df = df[df['language'] != 'french']
# Convert 'points' to numeric and calculate median
median_points = filtered_df['points'].astype(float).median()
print(f"Final Answer: {median_points}")