import pandas as pd

df = pd.read_csv('table.csv')

# Filter for 2008 Olympic Trials
trials_data = df[df['competition'] == 'olympic trials']

# Convert scores to float, handling 'dnq' and 'n/a'
def safe_float(x):
    if x == 'dnq' or x == 'n/a':
        return None
    return float(x)

trials_data['final_score'] = trials_data['final - score'].apply(safe_float)
trials_data['qualifying_score'] = trials_data['qualifying score'].apply(safe_float)

# Drop rows where scores are missing
valid_data = trials_data.dropna(subset=['final_score', 'qualifying_score'])

# Calculate Event Score Average
valid_data['event_score_avg'] = (valid_data['final_score'] + valid_data['qualifying_score']) / 2

# Find the event with the highest average
max_avg_event = valid_data.loc[valid_data['event_score_avg'].idxmax()]['event']

print(f"Final Answer: {max_avg_event}")