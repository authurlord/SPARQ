import pandas as pd

df = pd.read_csv('table.csv')

# Extract team name and number of titles
df['team'] = df['winner'].str.extract(r'([a-zA-Z\s]+)\s*\(\d+\)')
df['titles'] = df['winner'].str.extract(r'\((\d+)\)')

# Convert titles to integer
df['titles'] = pd.to_numeric(df['titles'])

# Find the team with the most titles
max_titles_team = df.loc[df['titles'].idxmax()]['team'].strip()

# Find the year when this team won their first title
first_title_year = df[df['team'] == max_titles_team]['year'].min()

# Get the winning coach for that year
winning_coach = df.loc[df['year'] == first_title_year, 'winning coach'].values[0]

print(f"Final Answer: {max_titles_team}, {winning_coach}")