import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out non-party rows
parties = df[df['Party'] != 'Total valid votes']
parties = parties[parties['Party'] != 'Total rejected ballots']
parties = parties[parties['Party'] != 'Turnout']
parties = parties[parties['Party'] != 'Eligible voters']

# Convert % to float
parties['%'] = pd.to_numeric(parties['%'])

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(parties['%'], labels=parties['Party'], autopct='%1.1f%%', startangle=90)
plt.title('Vote Share by Political Party')
plt.show()