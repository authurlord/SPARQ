import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'french'
filtered_df = df[df['language'] != 'french']
# Calculate median of points for the filtered data
median_points = filtered_df['points'].median()
print(f"Final Answer: {median_points}")