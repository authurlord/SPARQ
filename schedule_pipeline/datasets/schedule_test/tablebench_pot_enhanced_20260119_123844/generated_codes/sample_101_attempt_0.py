import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out non-candidate rows
candidates = df[df['Party'] != 'Total valid votes']
candidates = candidates[candidates['Party'] != 'Total rejected ballots']
candidates = candidates[candidates['Party'] != 'Turnout']
candidates = candidates[candidates['Party'] != 'Eligible voters']

# Convert percentage to float
candidates['%'] = candidates['%'].astype(float)

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(candidates['%'], labels=candidates['Party'], autopct='%1.1f%%', startangle=140)
plt.title('Vote Share by Political Party')
plt.show()