import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns and convert to numeric, ignoring any non-numeric values or formatting
columns_to_avg = ['basic elements', 'tumbling', 'stunts', 'tosses / pyramids']
df_numeric = df[columns_to_avg].apply(pd.to_numeric, errors='coerce')

# Calculate average skill score per team
df['average_skill_score'] = df_numeric.mean(axis=1)

# Find the team with the highest average skill score
top_team = df.loc[df['average_skill_score'].idxmax(), 'team name']

print(f"Final Answer: {top_team}")