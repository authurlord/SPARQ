import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select relevant columns for the stacked bar chart
data = df[['Club', 'Wins', 'Draws', 'Losses']]

# Convert to numeric (in case of any formatting issues)
data['Wins'] = pd.to_numeric(data['Wins'], errors='coerce')
data['Draws'] = pd.to_numeric(data['Draws'], errors='coerce')
data['Losses'] = pd.to_numeric(data['Losses'], errors='coerce')

# Drop rows with NaN values due to parsing errors
data = data.dropna()

# Plot stacked bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(data['Club'], data['Wins'], label='Wins', color='skyblue')
plt.bar(data['Club'], data['Draws'], bottom=data['Wins'], label='Draws', color='lightgreen')
plt.bar(data['Club'], data['Losses'], bottom=data['Wins'] + data['Draws'], label='Losses', color='salmon')

plt.xlabel('Club')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses by Club')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer is not a number or entity, but the question asks to draw the chart.
# Since the task is to draw, no final answer in the format is needed beyond the chart.
# However, per instructions, we must output the final answer format if applicable.
# As no specific value is requested, we return nothing in the final answer format.
Final Answer: