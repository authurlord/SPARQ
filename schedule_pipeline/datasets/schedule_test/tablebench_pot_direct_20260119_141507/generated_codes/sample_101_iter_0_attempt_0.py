import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter rows to include only candidate parties (exclude totals, rejected, turnout, etc.)
candidate_rows = df[df['Party'].str.contains('Conservative|Liberal|New Democratic|Green|Christian Heritage')]

# Extract party and percentage
vote_data = candidate_rows[['Party', '%']].dropna()

# Convert percentage to float
vote_data['%'] = vote_data['%'].str.replace('%', '').astype(float)

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(vote_data['%'], labels=vote_data['Party'], autopct='%1.1f%%', startangle=90)
plt.title('Vote Share of Candidates by Political Party')
plt.show()

print("Final Answer: pie_chart")