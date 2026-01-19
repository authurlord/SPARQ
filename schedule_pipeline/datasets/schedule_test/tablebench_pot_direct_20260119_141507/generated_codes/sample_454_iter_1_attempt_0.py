import pandas as pd

df = pd.read_csv('table.csv')

# Identify the episode with unusually high viewers
max_viewers = df['viewers'].max()
outlier_episode = df[df['viewers'] == max_viewers]['title'].values[0]

# Identify unusual timeslot (not in the standard 8:30 pm - 9:30 pm range)
standard_timeslot = '8:30 pm - 9:30 pm'
unusual_timeslot = df[df['timeslot'] != standard_timeslot]['timeslot'].values[0]

print(f"Final Answer: {outlier_episode}, {unusual_timeslot}")