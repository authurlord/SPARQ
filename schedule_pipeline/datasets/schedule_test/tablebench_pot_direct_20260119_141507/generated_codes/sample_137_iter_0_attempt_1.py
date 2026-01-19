import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Prepare data for plotting
counties = df['county']
obama_votes = df['obama']
mccain_votes = df['mccain']

# Create a bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(counties))

bars1 = plt.bar([i - bar_width/2 for i in index], obama_votes, bar_width, label='Obama', color='skyblue')
bars2 = plt.bar([i + bar_width/2 for i in index], mccain_votes, bar_width, label='McCain', color='salmon')

plt.xlabel('County')
plt.ylabel('Total Votes')
plt.title('Total Votes Received by Obama and McCain in Each County')
plt.xticks([i for i in index], counties, rotation=90)
plt.legend()

# Add value labels on top of bars
for bar1, bar2, votes1, votes2 in zip(bars1, bars2, obama_votes, mccain_votes):
    plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1, 
             f'{votes1}', ha='center', va='bottom', fontsize=9)
    plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1, 
             f'{votes2}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()