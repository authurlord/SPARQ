import pandas as pd

df = pd.read_csv('table.csv')

# Define month mapping for string comparison
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
    'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
    'November': 11, 'December': 12
}

# Extract month from Season Premiere
df['premiere_month'] = df['Season Premiere'].str.extract(r'(\w+)')
df['premiere_month_num'] = df['premiere_month'].map(month_map)

# Filter: episodes >= 40 AND premiere month between October and December
filtered_df = df[
    (df['Episodes'] >= 40) & 
    (df['premiere_month_num'] >= 10) & 
    (df['premiere_month_num'] <= 12)
]

# Calculate average episodes for filtered seasons
average_episodes = filtered_df['Episodes'].mean()

print(f"Final Answer: {average_episodes:.1f}")