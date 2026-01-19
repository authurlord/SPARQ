import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'transparency international'
transparency_data = df[df['organization'] == 'transparency international']
# Calculate the average rank
average_rank = transparency_data['rank'].astype(int).mean()
print(f"Final Answer: {average_rank:.1f}")