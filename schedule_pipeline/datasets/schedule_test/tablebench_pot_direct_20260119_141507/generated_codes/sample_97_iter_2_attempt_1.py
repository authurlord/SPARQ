import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter top 8 teams (rank 1 to 8)
top_8 = df[df['rank'].astype(int).between(1, 8)]

# Extract the relevant columns
data = top_8[['Won', 'Drawn', 'Lost']]

# Create a stacked bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(top_8['Club'], data.sum(axis=1), color=['skyblue', 'lightgreen', 'lightcoral'], edgecolor='black')

# Add labels and title
plt.xlabel('Club')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses for Top 8 Teams')

# Add value labels on top of each segment
for i, row in top_8.iterrows():
    wins, drawn, lost = row['Won'], row['Drawn'], row['Lost']
    plt.text(i, wins, f'{wins}', ha='center', va='bottom', fontsize=10, color='black')
    plt.text(i, wins + drawn, f'{drawn}', ha='center', va='bottom', fontsize=10, color='black')
    plt.text(i, wins + drawn + lost, f'{lost}', ha='center', va='bottom', fontsize=10, color='black')

# Improve layout
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()