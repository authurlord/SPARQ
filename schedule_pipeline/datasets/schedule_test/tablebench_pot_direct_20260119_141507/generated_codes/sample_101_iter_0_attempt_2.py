import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter rows where 'Party' is not a summary row (exclude total, rejected, turnout, etc.)
candidate_rows = df[df['Party'].str.contains('Conservative|Liberal|New Democratic|Green|Christian Heritage', case=False, na=False)]

# Extract party and percentage for plotting
party_votes = candidate_rows[['Party', '%']].dropna()
party_votes = party_votes[party_votes['%'].str.contains(r'^\d+(\.\d+)?$', na=False)]  # Ensure percentage is numeric

# Convert percentage to float
party_votes['%'] = party_votes['%'].astype(float)

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(party_votes['%'], labels=party_votes['Party'], autopct='%1.1f%%', startangle=90)
plt.title('Vote Share of Candidates by Political Party')
plt.show()

print(f"Final Answer: pie_chart")