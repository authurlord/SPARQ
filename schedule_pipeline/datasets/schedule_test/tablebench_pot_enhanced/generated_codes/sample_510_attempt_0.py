import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'english'
non_english_data = df[df['language'] != 'english']
# Calculate average points for non-English entries
avg_points = non_english_data['points'].mean()
print(f"Final Answer: {avg_points:.1f}")