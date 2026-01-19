import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Create a new column for labeling seasons (using League and League.1)
df['Season'] = df['League'] + ' ' + df['League.1']

# Set up the stacked bar chart
plt.figure(figsize=(12, 6))
plt.bar(df['Season'], df['Win'], label='Wins', color='green')
plt.bar(df['Season'], df['Lose'], label='Losses', color='red', bottom=df['Win'])

# Add labels and title
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.title('Wins and Losses by Season')
plt.xticks(rotation=45)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()