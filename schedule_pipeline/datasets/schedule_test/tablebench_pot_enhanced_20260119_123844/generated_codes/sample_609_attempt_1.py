import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Kyrgyzstan
kyrgyzstan_data = df[df['Unnamed: 0'] == 'kyrgyzstan']
total_withdrawal = float(kyrgyzstan_data['total freshwater withdrawal'].values[0])
domestic_use_percentage = 39 / 100
domestic_use = total_withdrawal * domestic_use_percentage
print(f"Final Answer: {domestic_use:.2f}")