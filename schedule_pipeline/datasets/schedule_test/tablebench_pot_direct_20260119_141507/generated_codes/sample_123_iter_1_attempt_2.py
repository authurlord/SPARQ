import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and prepare data for plotting
# Select only relevant columns
data = df[['Club', 'Wins', 'Draws', 'Losses']]

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(data['Club'], data['Wins'], label='Wins', color='skyblue')
plt.bar(data['Club'], data['Draws'], bottom=data['Wins'], label='Draws', color='lightgreen')
plt.bar(data['Club'], data['Losses'], bottom=data['Wins'] + data['Draws'], label='Losses', color='salmon')

# Customize labels and title
plt.xlabel('Club')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses by Club')
plt.legend()

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()

# Display the plot
plt.show()