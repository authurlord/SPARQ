import pandas as pd

df = pd.read_csv('table.csv')

# Select the relevant columns for skill scores
columns = ['basic elements', 'tumbling', 'stunts', 'tosses / pyramids']
df_scores = df[columns].apply(pd.to_numeric, errors='coerce')

# Calculate average skill score per team
df_scores['average_skill_score'] = df_scores.mean(axis=1)

# Find the team with the highest average skill score
max_score_team = df.loc[df_scores['average_skill_score'].idxmax(), 'team name']

print(f"Final Answer: {max_score_team}")