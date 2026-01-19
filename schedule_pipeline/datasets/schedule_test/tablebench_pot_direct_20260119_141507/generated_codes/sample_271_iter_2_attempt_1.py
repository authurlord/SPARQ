import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract month from Season Premiere using regex
df['month'] = df['Season Premiere'].str.extract(r'(\w+) (\d+)', expand=False)[0].str.strip()
df['month_num'] = df['month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
})

# Filter: episodes >= 40 and month between 10 and 12
filtered_df = df[(df['Episodes'] >= 40) & (df['month_num'] >= 10)]

# Calculate average episodes
average_episodes = filtered_df['Episodes'].mean()

print(f"Final Answer: {average_episodes:.1f}")