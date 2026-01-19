import pandas as pd

df = pd.read_csv('table.csv')

# Filter for 'all around' event in 'olympic trials' and 'us championships' in 2008
trials_row = df[(df['event'] == 'all around') & (df['competition'] == 'olympic trials')]
championships_row = df[(df['event'] == 'all around') & (df['competition'] == 'us championships')]

# Extract qualifying scores
qualifying_score_trials = float(trials_row['qualifying score'].values[0])
qualifying_score_championships = float(championships_row['qualifying score'].values[0])

# Calculate the change
change = qualifying_score_championships - qualifying_score_trials

print(f"Final Answer: {change:.1f}")