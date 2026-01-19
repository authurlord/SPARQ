import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Egypt and Morocco
egypt_growth = df[df['country (or dependent territory)'] == 'egypt']['average relative annual growth (%)'].values[0]
morocco_growth = df[df['country (or dependent territory)'] == 'morocco']['average relative annual growth (%)'].values[0]
# Calculate the difference
difference = float(egypt_growth) - float(morocco_growth)
print(f"Final Answer: {difference:.2f}")