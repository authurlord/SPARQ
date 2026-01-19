import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for Kyrgyzstan
kyrgyzstan_row = df[df['Unnamed: 0'] == 'kyrgyzstan']
# Extract total freshwater withdrawal
total_withdrawal = float(kyrgyzstan_row['total freshwater withdrawal'].values[0])
# Calculate domestic use (39% of total)
domestic_use = total_withdrawal * 0.39
print(f"Final Answer: {domestic_use:.2f}")