import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter only running events (by event name)
running_events = ['60 metres', '200 metres', '400 metres', '800 metres', '1500 metres', '3000 metres']

# Filter rows where event is in running_events
filtered_df = df[df['Event'].isin(running_events)]

# Extract event length (remove 'metres' and convert to int)
def get_event_length(event):
    return int(event.replace(' metres', ''))

filtered_df['length'] = filtered_df['Event'].apply(get_event_length)

# Extract winning time (from Gold.1 column)
# Gold.1 may contain times like '7.17', '22.41', '2:00.42'
def parse_time(time_str):
    if isinstance(time_str, str):
        if ':' in time_str:
            # Format X:Y.Z
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds_part = parts[1]
            seconds = float(seconds_part.split('.')[0]) + float(seconds_part.split('.')[1]) / 100
            return minutes * 60 + seconds
        else:
            # Format X.Y
            return float(time_str)
    return None

# Apply parsing to Gold.1 column
filtered_df['winning_time_seconds'] = filtered_df['Gold.1'].apply(parse_time)

# Drop rows where time is NaN
filtered_df = filtered_df.dropna(subset=['winning_time_seconds'])

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['length'], filtered_df['winning_time_seconds'], color='blue', alpha=0.7)
plt.title('Relationship between Event Length and Winning Time (Running Events)')
plt.xlabel('Event Length (metres)')
plt.ylabel('Winning Time (seconds)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Final answer is not a number, but we are asked to draw the plot — so no final print of answer
# However, since the question says "draw a scatter plot", we just generate the plot.