import pandas as pd

df = pd.read_csv('table.csv')
# Convert the score columns to numeric, handling potential non-numeric entries
score_columns = ['basic elements', 'tumbling', 'stunts', 'tosses / pyramids']
df[score_columns] = df[score_columns].apply(pd.to_numeric, errors='coerce')

# Calculate the average skill score for each team
df['Average Skill Score'] = df[score_columns].mean(axis=1)

# Find the team with the highest average skill score
highest_scorer = df.loc[df['Average Skill Score'].idxmax(), 'team name']

print(f"Final Answer: {highest_scorer}")