import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where language is not 'english'
non_english_entries = df[df['language'] != 'english']
# Calculate the average points for these entries
average_points = non_english_entries['points'].mean()
print(f"Final Answer: {average_points:.1f}")