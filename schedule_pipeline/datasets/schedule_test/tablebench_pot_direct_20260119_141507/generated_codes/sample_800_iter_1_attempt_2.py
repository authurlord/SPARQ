import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where language is not French
filtered_df = df[df['language'] != 'french']
# Calculate median of points for non-French languages
median_points = filtered_df['points'].median()
print(f"Final Answer: {median_points}")