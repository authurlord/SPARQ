import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'english'
non_english_data = df[df['language'] != 'english']
# Calculate the average points for non-English entries
average_points = non_english_data['points'].mean()
print(f"Final Answer: {average_points:.1f}")