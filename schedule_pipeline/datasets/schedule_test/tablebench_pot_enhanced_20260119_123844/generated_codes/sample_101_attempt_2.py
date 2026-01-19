import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter out non-candidate rows (keep only actual parties)
parties = df[df['Candidate'] != 'Total valid votes']
parties = parties[parties['Candidate'] != 'Total rejected ballots']
parties = parties[parties['Candidate'] != 'Turnout']
parties = parties[parties['Candidate'] != 'Eligible voters']

# Convert % to float
parties['%'] = pd.to_numeric(parties['%'])

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(parties['%'], labels=parties['Party'], autopct='%1.1f%%', startangle=90)
plt.title('Vote Share by Party')
plt.show()