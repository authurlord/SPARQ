import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter only running events (exclude walking)
running_events = ['60 metres', '200 metres', '400 metres', '800 metres', '1500 metres', '3000 metres']
filtered_df = df[df['Event'].isin(running_events)]

# Extract event length (in metres) and winning time (Gold column)
event_lengths = [int(event.split()[0]) for event in filtered_df['Event']]
winning_times = filtered_df['Gold'].astype(str).str.replace(' ', '').str.replace(':', '.').astype(float)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(event_lengths, winning_times, color='blue')
plt.title('Scatter Plot of Event Length vs Winning Time (Running Events)')
plt.xlabel('Length of Event (metres)')
plt.ylabel('Winning Time (seconds)')
plt.grid(True)
plt.show()