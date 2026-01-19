import pandas as pd
from datetime import datetime

df = pd.read_csv('table.csv')

# Filter for locomotives built in 1938
df_1938 = df[df['Date Built'].str.contains('1938', na=False)]

# Convert Date Built and Date Withdrawn to datetime objects
def parse_date(date_str):
    # Handle common formats like "January 1938", "December 1938"
    try:
        return datetime.strptime(date_str, "%B %Y")
    except ValueError:
        return None

df_1938['Date Built'] = df_1938['Date Built'].apply(parse_date)
df_1938['Date Withdrawn'] = df_1938['Date Withdrawn'].apply(parse_date)

# Calculate duration in years
df_1938['Duration (years)'] = (df_1938['Date Withdrawn'] - df_1938['Date Built']).dt.days / 365.25

# Find the maximum duration
max_years = df_1938['Duration (years)'].max()

print(f"Final Answer: {int(max_years)}")