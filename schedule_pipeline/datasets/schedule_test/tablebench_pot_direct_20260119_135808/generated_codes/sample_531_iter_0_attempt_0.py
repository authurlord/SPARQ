import pandas as pd

df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric
columns_to_average = ['basic elements', 'tumbling', 'stunts', 'tosses / pyramids']
df[columns_to_average] = df[columns_to_average].apply(pd.to_numeric)

# Calculate the average skill score for each team
df['Average Skill Score'] = df[columns_to_average].mean(axis=1)

# Find the team with the highest average skill score
top_team = df.loc[df['Average Skill Score'].idxmax(), 'team name']

print(f"Final Answer: {top_team}")