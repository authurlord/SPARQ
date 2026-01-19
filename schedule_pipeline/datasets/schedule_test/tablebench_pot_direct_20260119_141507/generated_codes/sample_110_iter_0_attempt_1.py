import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter only running events (exclude jumping and walking events)
running_events = [
    '60 metres', '200 metres', '400 metres', '800 metres', '1500 metres', '3000 metres', '60 metres hurdles'
]

# Filter data for running events
filtered_df = df[df['Event'].isin(running_events)]

# Extract event length (in meters) from the event name
def extract_length(event):
    # Remove "metres" or "hurdles" and convert to number
    if 'metres' in event:
        return int(event.replace('metres', '').strip())
    elif 'hurdles' in event:
        return 60
    return None

filtered_df['length'] = filtered_df['Event'].apply(extract_length)

# Parse winning times (Gold column) into seconds
def parse_time(time_str):
    if ':' in time_str:
        minutes, seconds = time_str.split(':')
        return float(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

filtered_df['winning_time'] = filtered_df['Gold'].apply(parse_time)

# Drop rows where length is null
filtered_df = filtered_df.dropna(subset=['length', 'winning_time'])

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['length'], filtered_df['winning_time'], alpha=0.7)
plt.title('Scatter Plot of Event Length vs Winning Time (Running Events)')
plt.xlabel('Length of Event (meters)')
plt.ylabel('Winning Time (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final answer is not a value but a plot — however, the question asks to "draw" the plot.
# Since no specific value is requested, we just generate the plot.
# Final Answer is not applicable as per format, but the task was to draw the plot.
# However, the final answer format requires a list of names/numbers.
# Since no named answer is requested, we interpret that the task is completed by plotting.
# But per instruction, we must output in Final Answer: ...
# Since there's no quantitative answer, we return a placeholder based on requirement.

Final Answer: plot