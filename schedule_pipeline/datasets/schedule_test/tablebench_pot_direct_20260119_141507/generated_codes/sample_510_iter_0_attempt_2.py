import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where language is not 'english'
non_english_points = df[df['language'] != 'english']['points']
# Calculate the average (mean) of points
average_points = non_english_points.mean()
print(f"Final Answer: {average_points:.1f}")