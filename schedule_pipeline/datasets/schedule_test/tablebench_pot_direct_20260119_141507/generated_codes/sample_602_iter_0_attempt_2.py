import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for Glasgow International
glasgow_row = df[df['airport'] == 'glasgow international']
total_passengers = glasgow_row['total passengers'].values[0]
# Increase by 15%
new_passengers = total_passengers * 1.15
print(f"Final Answer: {int(new_passengers)}")