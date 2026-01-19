import pandas as pd

df = pd.read_csv('table.csv')

# Filter for 2008 Olympic Trials
trial_data = df[df['competition'] == 'olympic trials']

# Calculate Event Score Average (average of final score and qualifying score)
trial_data['event_score_average'] = (trial_data['final - score'].astype(float) + 
                                     trial_data['qualifying score'].astype(float)) / 2

# Find the event with the highest average
highest_avg_event = trial_data.loc[trial_data['event_score_average'].idxmax(), 'event']

print(f"Final Answer: {highest_avg_event}")