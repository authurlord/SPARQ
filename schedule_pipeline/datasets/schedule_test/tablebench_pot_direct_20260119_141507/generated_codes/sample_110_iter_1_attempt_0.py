import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Define running events (excluding hurdles, walk, jumps, etc.)
running_events = [
    '60 metres', '200 metres', '400 metres', '800 metres', '1500 metres', '3000 metres'
]

# Filter rows where event is in running_events
filtered_df = df[df['Event'].isin(running_events)]

# Function to convert time string to seconds
def parse_time(time_str):
    if isinstance(time_str, str):
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1].split('.')[0]) + float(parts[1].split('.')[1]) / 100
            return minutes * 60 + seconds
        else:
            return float(time_str)
    return 0

# Extract event length (in meters) and winning time (in seconds)
lengths = []
gold_times = []

for index, row in filtered_df.iterrows():
    event = row['Event']
    length = int(event.split()[0])  # e.g., '60 metres' → 60
    gold_time = parse_time(row['Gold'])
    lengths.append(length)
    gold_times.append(gold_time)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(lengths, gold_times, color='blue')
plt.title('Scatter Plot: Event Length (meters) vs Winning Time (seconds)')
plt.xlabel('Event Length (meters)')
plt.ylabel('Winning Time (seconds)')
plt.grid(True)
plt.show()

# Final Answer is not a number or entity, but the plot is generated.
# Since the question asks to draw the scatter plot, we don't return a value.
# However, per instruction, we must output final answer in required format.
# As no specific value is requested, we just confirm the plot was generated.
Final Answer: scatter_plot_generated