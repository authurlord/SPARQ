import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter rows to include only candidate vote shares (exclude totals and non-candidate rows)
candidate_data = df[df['Party'].notna() & (df['%'].notna()) & (df['Candidate'].notna())]

# Extract party and percentage
parties = candidate_data['Party']
votes_percent = candidate_data['%'].astype(float)

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(votes_percent, labels=parties, autopct='%1.1f%%', startangle=90)
plt.title('Vote Share of Candidates by Political Party')
plt.show()

print("Final Answer: pie_chart")