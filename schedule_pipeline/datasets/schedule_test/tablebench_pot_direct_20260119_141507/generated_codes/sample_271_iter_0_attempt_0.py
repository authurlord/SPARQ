import pandas as pd

df = pd.read_csv('table.csv')

# Filter seasons with at least 40 episodes and premiered between October and December
filtered_df = df[
    (df['Episodes'] >= 40) &
    (df['Season Premiere'].str.contains(r'October|November|December', case=False))
]

# Calculate the average number of episodes for the filtered seasons
average_episodes = filtered_df['Episodes'].mean()

print(f"Final Answer: {average_episodes:.1f}")