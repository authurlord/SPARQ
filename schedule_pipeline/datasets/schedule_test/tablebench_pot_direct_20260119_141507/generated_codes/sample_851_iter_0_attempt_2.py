import pandas as pd

df = pd.read_csv('table.csv')
# Extract the average relative annual growth (%) for Egypt and Morocco
egypt_growth = float(df[df['country (or dependent territory)'] == 'egypt']['average relative annual growth (%)'])
morocco_growth = float(df[df['country (or dependent territory)'] == 'morocco']['average relative annual growth (%)'])

# Calculate the difference
difference = egypt_growth - morocco_growth
print(f"Final Answer: {difference:.2f}")