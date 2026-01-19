import pandas as pd

df = pd.read_csv('table.csv')

# Select only the relevant columns for skill scores
columns_to_average = ['basic elements', 'tumbling', 'stunts', 'tosses / pyramids']
# Convert the string values to float (removing any parentheses or non-numeric parts)
df_clean = df.copy()
for col in columns_to_average:
    # Remove any text like '(13)' and convert to float
    df_clean[col] = df_clean[col].str.extract(r'(\d+\.?\d*)').astype(float)

# Calculate average skill score per team
df_clean['average_skill_score'] = df_clean[columns_to_average].mean(axis=1)

# Find the team with the highest average skill score
top_team = df_clean.loc[df_clean['average_skill_score'].idxmax(), 'team name']

print(f"Final Answer: {top_team}")