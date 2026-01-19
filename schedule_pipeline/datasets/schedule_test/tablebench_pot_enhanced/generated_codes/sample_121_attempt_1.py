import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Winners' and 'Runner-ups' to numeric, treating '-' as 0
df['Winners'] = pd.to_numeric(df['Winners'], errors='coerce').fillna(0).astype(int)
df['Runner-ups'] = pd.to_numeric(df['Runner-ups'], errors='coerce').fillna(0).astype(int)

# Set up the bar chart
clubs = df['Club']
win_counts = df['Winners']
runner_up_counts = df['Runner-ups']

x = range(len(clubs))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x, win_counts, width, label='Wins', color='blue')
plt.bar([i + width for i in x], runner_up_counts, width, label='Runner-ups', color='orange')

plt.xlabel('Clubs')
plt.ylabel('Number of Appearances')
plt.title('Number of Wins and Runner-up Positions by Club')
plt.xticks([i + width / 2 for i in x], clubs, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer: Bar chart displayed
print("Final Answer: Bar chart displayed")