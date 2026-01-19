import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the relevant columns
votes_obama = df['obama']
votes_mccain = df['mccain']
counties = df['county']

# Create a bar chart
plt.figure(figsize=(12, 6))
x = range(len(counties))
width = 0.35

plt.bar([i - width/2 for i in x], votes_obama, width, label='Obama', color='skyblue')
plt.bar([i + width/2 for i in x], votes_mccain, width, label='McCain', color='salmon')

plt.xlabel('County')
plt.ylabel('Total Votes')
plt.title('Total Votes Received by Obama and McCain in Each County')
plt.xticks([i for i in x], counties, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()