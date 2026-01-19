import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Competition is 'World Championships' and Position is 5th or higher
# Position values like '5th (semis)', '4th (heats)' — we need to check if it's 5th or above
filtered_df = df[(df['Competition'] == 'World Championships') & 
                 (df['Position'].str.contains(r'^(5th|4th|3rd|2nd|1st)', case=False, na=False))]

# Extract the Notes column (contains times like '1:45.61')
times = filtered_df['Notes'].str.extract(r'(\d+:\d+\.\d+)').dropna()

# Convert time string to seconds
def time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

fastest_time_seconds = min(time_to_seconds(t) for t in times)

# Convert back to mm:ss format for readability (optional, but the question asks for the time)
fastest_time = f"{int(fastest_time_seconds // 60):02d}:{fastest_time_seconds % 60:.2f}"

print(f"Final Answer: {fastest_time}")