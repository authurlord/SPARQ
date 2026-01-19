import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'transparency international'
transparency_df = df[df['organization'] == 'transparency international']
# Calculate the average rank
avg_rank = transparency_df['rank'].mean()
print(f"Final Answer: {avg_rank:.1f}")