import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where venue is Waverley Park and crowd > 40000
filtered_df = df[(df['venue'] == 'waverley park') & (df['crowd'] > 40000)]
# Find the row with the highest margin
highest_margin_row = filtered_df.loc[filtered_df['margin'].idxmax()]
# Return the winners (the team with the highest margin)
final_answer = highest_margin_row['winners']
print(f"Final Answer: {final_answer}")