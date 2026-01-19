import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where class is 'gt1' and co-drivers contain 'peter kox'
filtered_df = df[(df['class'] == 'gt1') & (df['co - drivers'].str.contains('peter kox', case=False, na=False))]
# Extract the team name
team = filtered_df['team'].iloc[0]
print(f"Final Answer: {team}")