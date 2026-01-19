import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is not 'french'
non_french_df = df[df['language'] != 'french']
# Calculate median of points for non-french artists
median_points = non_french_df['points'].median()
print(f"Final Answer: {median_points}")