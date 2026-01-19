import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter out non-party rows (keep only actual parties)
party_data = df[df['Party'] != 'Total valid votes']
party_data = party_data[party_data['Party'] != 'Total rejected ballots']
party_data = party_data[party_data['Party'] != 'Turnout']
party_data = party_data[party_data['Party'] != 'Eligible voters']

# Convert percentage to float
party_data['%'] = pd.to_numeric(party_data['%'])

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(party_data['%'], labels=party_data['Party'], autopct='%1.1f%%', startangle=90)
plt.title('Vote Share by Political Party')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()