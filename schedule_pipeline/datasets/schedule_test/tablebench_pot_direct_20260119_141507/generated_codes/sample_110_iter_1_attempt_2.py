import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Filter only running events (exclude walking events)
running_events = [
    '60 metres', '200 metres', '400 metres', '800 metres', '1500 metres', '3000 metres'
]

# Filter rows where Event is in running_events
filtered_df = df[df['Event'].isin(running_events)]

# Extract event length (in metres) from the event name
def extract_length(event):
    match = re.search(r'(\d+) metres', event)
    return int(match.group(1)) if match else None

filtered_df['length'] = filtered_df['Event'].apply(extract_length)

# Convert winning time (Gold) from string to float
filtered_df['winning_time'] = pd.to_numeric(filtered_df['Gold'], errors='coerce')

# Drop rows where time conversion failed
filtered_df = filtered_df.dropna(subset=['winning_time'])

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['length'], filtered_df['winning_time'], color='blue')
plt.title('Scatter Plot: Event Length vs Winning Time (Running Events)')
plt.xlabel('Length of Event (metres)')
plt.ylabel('Winning Time (seconds)')
plt.grid(True)
plt.show()

# Final Answer is not a value but a plot, so we just print a confirmation
print("Final Answer: scatter_plot")