import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the vote counts for Obama and McCain
obama_votes = df['obama']
mccain_votes = df['mccain']

# Create a list of counties for the x-axis
counties = df['county']

# Combine into a DataFrame for plotting
votes_data = pd.DataFrame({
    'County': counties,
    'Obama': obama_votes,
    'McCain': mccain_votes
})

# Create a bar chart
plt.figure(figsize=(14, 8))
bar_width = 0.35
index = range(len(votes_data))

# Plot Obama and McCain votes
bars1 = plt.bar([i - bar_width/2 for i in index], votes_data['Obama'], bar_width, label='Obama', color='skyblue')
bars2 = plt.bar([i + bar_width/2 for i in index], votes_data['McCain'], bar_width, label='McCain', color='salmon')

# Labels and title
plt.xlabel('County')
plt.ylabel('Total Votes')
plt.title('Total Votes Received by Obama and McCain in Each County in New Mexico')
plt.xticks([i for i in index], votes_data['County'], rotation=45)

# Legend
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer is not a value but a chart, so we just output the required format
Final Answer: chart