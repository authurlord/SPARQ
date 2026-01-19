import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert Winners and Runner-ups to integers
df['Winners'] = pd.to_numeric(df['Winners'], errors='coerce')
df['Runner-ups'] = pd.to_numeric(df['Runner-ups'], errors='coerce')

# Create a new DataFrame for plotting
plot_data = df[['Club', 'Winners', 'Runner-ups']]
plot_data = plot_data.dropna()  # Remove any rows with NaN values

# Set up the bar chart
plt.figure(figsize=(12, 6))
x = plot_data['Club']
winners_values = plot_data['Winners']
runner_ups_values = plot_data['Runner-ups']

# Create grouped bar chart
bar_width = 0.35
index = range(len(x))
bars1 = plt.bar([i - bar_width/2 for i in index], winners_values, bar_width, label='Winners', color='skyblue')
bars2 = plt.bar([i + bar_width/2 for i in index], runner_ups_values, bar_width, label='Runner-ups', color='lightcoral')

# Customize the chart
plt.xlabel('Club')
plt.ylabel('Number of Times')
plt.title('Number of Times Each Club Has Won First and Second Place')
plt.xticks([i for i in index], x, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the chart
plt.show()