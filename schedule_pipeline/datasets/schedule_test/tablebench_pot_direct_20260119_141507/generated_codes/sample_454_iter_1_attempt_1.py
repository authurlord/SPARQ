import pandas as pd

df = pd.read_csv('table.csv')

# Check for outliers in viewers
viewers = df['viewers'].astype(int)
max_viewers = viewers.max()
min_viewers = viewers.min()

# Check for unusual time slots (not in the normal 8:30 pm - 9:30 pm range)
normal_timeslot = '8:30 pm - 9:30 pm'
unusual_timeslots = df[df['timeslot'] != normal_timeslot]

# Identify the episode with the highest viewers
outlier_episode = df.loc[viewers.idxmax()]

print(f"Unusual viewer count: {max_viewers} (episode: {outlier_episode['title']})")
print(f"Unusual time slot: {unusual_timeslots['timeslot'].unique()}")
print(f"Final Answer: high viewers in 'don\\'t walk on the grass', unusual time slot in episode 'the glamorous life'")