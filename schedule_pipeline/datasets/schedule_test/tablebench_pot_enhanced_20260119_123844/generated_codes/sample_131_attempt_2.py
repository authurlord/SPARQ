import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Set the league season as index for plotting
df.set_index('League.1', inplace=True)

# Create a stacked bar chart
df[['Win', 'Lose']].plot(kind='bar', stacked=True, figsize=(10, 6))

# Add labels and title
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.title('Wins and Losses by Season')
plt.xticks(rotation=45)
plt.legend(['Wins', 'Losses'])

# Show the plot
plt.tight_layout()
plt.show()