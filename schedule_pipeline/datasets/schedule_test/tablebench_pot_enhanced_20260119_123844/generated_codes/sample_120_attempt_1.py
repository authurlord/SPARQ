import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Viewers' to numeric, removing ' million' and converting to float
df['Viewers'] = df['Viewers'].str.replace(' million', '').astype(float)

# Extract the years and viewership data
years = df['Year'].astype(str)
viewers = df['Viewers']

# Create a waterfall chart
fig, ax = plt.subplots(figsize=(10, 6))

# Set the x-axis labels
x_pos = range(len(years))

# Calculate the cumulative sum for the waterfall effect
cumulative = [0] + list(viewers.cumsum())

# Plot bars
ax.bar(x_pos, viewers, color='skyblue', edgecolor='black')
ax.plot(x_pos, cumulative[:-1], color='red', linestyle='--', marker='o')

# Annotate values
for i, (v, y) in enumerate(zip(viewers, years)):
    ax.text(i, v + 0.1, f"{v}", ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Viewers (million)')
ax.set_title('Trend in Viewership of the Program (2007–2013)')
ax.set_xticks(x_pos)
ax.set_xticklabels(years)

# Show the plot
plt.tight_layout()
plt.show()

# Print final answer as requested
print("Final Answer: 6.574, 6.668, 6.487, 5.709, 6.337, 5.1, 5.5")