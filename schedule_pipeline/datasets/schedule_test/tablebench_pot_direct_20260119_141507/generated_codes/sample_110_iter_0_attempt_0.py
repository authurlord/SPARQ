import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter only running events (track running events)
running_events = ['60 metres', '200 metres', '400 metres', '800 metres', '1500 metres', '3000 metres']

# Filter rows for running events
filtered_df = df[df['Event'].isin(running_events)]

# Convert winning times to seconds
def time_to_seconds(time_str):
    if isinstance(time_str, str):
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds_part = parts[1]
            seconds = float(seconds_part.split('.')[0]) + float(seconds_part.split('.')[1]) / 100
            return minutes * 60 + seconds
        else:
            # e.g., '7.17'
            return float(time_str)
    return 0

filtered_df['winning_time_seconds'] = filtered_df['Gold'].apply(time_to_seconds)

# Extract event length in meters
event_lengths = {
    '60 metres': 60,
    '200 metres': 200,
    '400 metres': 400,
    '800 metres': 800,
    '1500 metres': 1500,
    '3000 metres': 3000
}

# Map event to length
filtered_df['length_meters'] = filtered_df['Event'].map(event_lengths)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['length_meters'], filtered_df['winning_time_seconds'], alpha=0.8)
plt.title('Scatter Plot of Event Length vs Winning Time (Running Events)')
plt.xlabel('Length of Event (meters)')
plt.ylabel('Winning Time (seconds)')
plt.grid(True)
plt.show()