import pandas as pd
from datetime import datetime

df = pd.read_csv('table.csv')

# Extract month from Season Premiere (e.g., "October 4, 2006" → 10)
def get_month(date_str):
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    month_name = date_str.split()[0]
    return month_map[month_name]

# Apply the function to get month values
df['premiere_month'] = df['Season Premiere'].apply(get_month)

# Filter: episodes >= 40 and premiere month between 10 and 12 (inclusive)
filtered_df = df[(df['Episodes'] >= 40) & (df['premiere_month'] >= 10)]

# Calculate average episodes for filtered seasons
average_episodes = filtered_df['Episodes'].mean()

print(f"Final Answer: {average_episodes:.1f}")