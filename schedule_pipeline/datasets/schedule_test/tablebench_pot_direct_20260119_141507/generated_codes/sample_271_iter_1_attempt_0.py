import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where episodes >= 40 and premiere month is between October and December
df_filtered = df[
    (df['Episodes'] >= 40) &
    (df['Season Premiere'].str.startswith('October') | 
     df['Season Premiere'].str.startswith('November') | 
     df['Season Premiere'].str.startswith('December'))
]

# Calculate the average number of episodes for the filtered seasons
average_episodes = df_filtered['Episodes'].mean()

print(f"Final Answer: {average_episodes:.1f}")