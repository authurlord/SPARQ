import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Prepare data for bar chart: county names and votes for Obama and McCain
data = df[['county', 'obama', 'mccain']]
data['total_votes'] = data['obama'] + data['mccain']

# Create a bar chart comparing Obama and McCain votes per county
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = range(len(data))

# Bar positions for Obama and McCain
bars_obama = plt.bar([i - bar_width/2 for i in index], data['obama'], bar_width, label='Obama', color='skyblue')
bars_mccain = plt.bar([i + bar_width/2 for i in index], data['mccain'], bar_width, label='McCain', color='salmon')

# Labels and title
plt.xlabel('County')
plt.ylabel('Number of Votes')
plt.title('Total Votes Received by Obama and McCain in Each County in New Mexico')
plt.xticks([i for i in index], data['county'], rotation=45, ha='right')
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

print(f"Final Answer: Bar chart created successfully")