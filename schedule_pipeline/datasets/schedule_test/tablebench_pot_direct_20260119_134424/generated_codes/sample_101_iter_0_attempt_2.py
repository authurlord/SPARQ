import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter out non-candidate rows
candidate_data = df[df['Candidate'] != 'Total valid votes']
candidate_data = candidate_data[candidate_data['Candidate'] != 'Total rejected ballots']
candidate_data = candidate_data[candidate_data['Candidate'] != 'Turnout']
candidate_data = candidate_data[candidate_data['Candidate'] != 'Eligible voters']

# Convert '%' column to float for plotting
candidate_data['%'] = candidate_data['%'].astype(float)

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(candidate_data['%'], labels=candidate_data['Party'], autopct='%1.1f%%', startangle=140)
plt.title('Vote Share by Political Party')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()