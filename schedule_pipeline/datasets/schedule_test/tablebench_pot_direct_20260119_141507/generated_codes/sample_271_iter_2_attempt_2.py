import pandas as pd
from datetime import datetime

df = pd.read_csv('table.csv')

# Extract month from Season Premiere (e.g., "October" -> 10)
def get_month(premiere):
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    month_str = premiere.split(',')[0].strip()
    return month_map.get(month_str)

# Apply the function to create a new column for month
df['premiere_month'] = df['Season Premiere'].apply(get_month)

# Filter: episodes >= 40 AND premiere month between October (10) and December (12)
filtered_df = df[(df['Episodes'] >= 40) & (df['premiere_month'] >= 10)]

# Calculate average episodes for filtered seasons
average_episodes = filtered_df['Episodes'].mean()

print(f"Final Answer: {average_episodes:.1f}")