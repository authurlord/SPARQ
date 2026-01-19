import pandas as pd

df = pd.read_csv('table.csv')
# Convert the score columns to float
score_columns = ['basic elements', 'tumbling', 'stunts', 'tosses / pyramids']
df[score_columns] = df[score_columns].astype(float)

# Calculate average skill score for each team
df['Average Skill Score'] = df[score_columns].mean(axis=1)

# Find the team with the highest average skill score
top_team = df.loc[df['Average Skill Score'].idxmax(), 'team name']

print(f"Final Answer: {top_team}")