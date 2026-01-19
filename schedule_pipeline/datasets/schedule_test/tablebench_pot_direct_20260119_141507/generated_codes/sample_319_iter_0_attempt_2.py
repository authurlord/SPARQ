import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert time string to total seconds for comparison
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Filter for Stockholm Marathon with time < 2:35:00
filtered = df[
    (df['Venue'] == 'Stockholm, Sweden') &
    (df['Event'] == 'Marathon')
]

# Convert time to seconds and filter
filtered['time_seconds'] = filtered['Notes'].apply(time_to_seconds)
filtered = filtered[filtered['time_seconds'] < 2*3600 + 35*60 + 0]

# Extract the year
years = filtered['Year'].dropna().tolist()

# If there are multiple years, we return all; otherwise, just one
final_years = list(set(years))  # Remove duplicates if any

print(f"Final Answer: {', '.join(map(str, final_years))}")