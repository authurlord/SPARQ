import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'english'
non_english_points = df[df['language'] != 'english']['points']
# Calculate the average points
average_points = non_english_points.mean()
print(f"Final Answer: {average_points:.1f}")