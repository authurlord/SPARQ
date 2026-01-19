import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where episodes >= 40 and premiere date is between October and December
filtered_df = df[
    (df['Episodes'] >= 40) &
    (df['Season Premiere'].str.contains('October|November|December', case=False))
]

# Calculate average episodes for the filtered seasons
average_episodes = filtered_df['Episodes'].mean()
print(f"Final Answer: {average_episodes:.1f}")