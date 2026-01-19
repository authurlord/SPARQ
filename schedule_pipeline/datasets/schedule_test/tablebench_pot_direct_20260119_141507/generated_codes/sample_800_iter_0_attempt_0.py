import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where language is not 'french'
other_languages = df[df['language'] != 'french']
# Calculate median of points for these rows
median_points = other_languages['points'].median()
print(f"Final Answer: {median_points}")