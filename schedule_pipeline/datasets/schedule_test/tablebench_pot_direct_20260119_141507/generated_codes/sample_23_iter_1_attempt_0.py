import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'transparency international'
ti_rows = df[df['organization'] == 'transparency international']
# Calculate average rank of these rows
average_rank = ti_rows['rank'].mean()
print(f"Final Answer: {average_rank:.1f}")