import pandas as pd

df = pd.read_csv('table.csv')
# Extract the row for Spanish speakers
spanish_row = df[df['language'] == 'spanish']
# Find the municipality with the highest value
max_spanish_municipality = spanish_row.iloc[0].iloc[1:].idxmax()
print(f"Final Answer: {max_spanish_municipality}")