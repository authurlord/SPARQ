import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '400 m'
filtered_df = df[df['Event'] == '400 m']
# Sort by Notes (time) ascending to get the fastest time
filtered_df_sorted = filtered_df.sort_values(by='Notes').reset_index(drop=True)
# The first row (fastest time) is the personal best
personal_best_year = filtered_df_sorted.iloc[0]['Year']
print(f"Final Answer: {personal_best_year}")