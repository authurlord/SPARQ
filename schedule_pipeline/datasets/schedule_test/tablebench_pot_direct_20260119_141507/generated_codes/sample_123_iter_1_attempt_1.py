import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and extract Wins, Draws, Losses as integers
# Remove '+' and convert to int
df['Wins'] = df['Wins'].str.replace('+', '').astype(int)
df['Draws'] = df['Draws'].str.replace('+', '').astype(int)
df['Losses'] = df['Losses'].str.replace('+', '').astype(int)

# Select relevant columns for plotting
data = df[['Club', 'Wins', 'Draws', 'Losses']]

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(data['Club'], data['Wins'], label='Wins', color='skyblue')
plt.bar(data['Club'], data['Draws'], bottom=data['Wins'], label='Draws', color='lightgreen')
plt.bar(data['Club'], data['Losses'], bottom=data['Wins'] + data['Draws'], label='Losses', color='salmon')

plt.xlabel('Club')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses by Team')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()