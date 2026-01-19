import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for 2008 Olympic Trials
trials_data = df[(df['year'] == '2008') & (df['competition'] == 'olympic trials')]

# Drop rows where final score is 'dnq'
trials_data = trials_data[trials_data['final - score'] != 'dnq']

# Convert final and qualifying scores to float
trials_data['final - score'] = pd.to_numeric(trials_data['final - score'])
trials_data['qualifying score'] = pd.to_numeric(trials_data['qualifying score'])

# Calculate Event Score Average
trials_data['event_score_avg'] = (trials_data['final - score'] + trials_data['qualifying score']) / 2

# Find the event with the highest average
max_avg_event = trials_data.loc[trials_data['event_score_avg'].idxmax()]['event']

print(f"Final Answer: {max_avg_event}")