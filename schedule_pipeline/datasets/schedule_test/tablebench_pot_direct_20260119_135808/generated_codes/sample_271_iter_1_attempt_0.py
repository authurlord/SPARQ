import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Episodes' to integer
df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')

# Extract month from 'Season Premiere' and convert to string
df['Month'] = df['Season Premiere'].str.extract(r'(\w+)', expand=False)

# Define months of interest
months_of_interest = ['October', 'November', 'December']

# Filter data: episodes >= 40 and premiere month in October-December
filtered_df = df[(df['Episodes'] >= 40) & (df['Month'].isin(months_of_interest))]

# Calculate average episodes for filtered seasons
if not filtered_df.empty:
    avg_episodes = filtered_df['Episodes'].mean()
else:
    avg_episodes = 0

print(f"Final Answer: {avg_episodes:.1f}")