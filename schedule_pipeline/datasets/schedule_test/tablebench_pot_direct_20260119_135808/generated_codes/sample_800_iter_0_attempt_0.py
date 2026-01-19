import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'french'
non_french_data = df[df['language'] != 'french']
# Calculate median of points for non-French languages
median_points = non_french_data['points'].median()
print(f"Final Answer: {median_points}")