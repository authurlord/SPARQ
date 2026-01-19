import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric, handling any non-numeric values
columns_to_convert = ['basic elements', 'tumbling', 'stunts', 'tosses / pyramids']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Calculate the average skill score for each team
df['Average Skill Score'] = df[columns_to_convert].mean(axis=1)

# Find the team with the highest average skill score
highest_scoring_team = df.loc[df['Average Skill Score'].idxmax(), 'team name']

print(f"Final Answer: {highest_scoring_team}")