import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'transparency international'
ti_indices = df[df['organization'] == 'transparency international']
# Calculate the average rank
average_rank = ti_indices['rank'].mean()
print(f"Final Answer: {average_rank:.1f}")