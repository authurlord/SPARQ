import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Pin (Tip)' column to numeric
df['Pin (Tip)'] = pd.to_numeric(df['Pin (Tip)'])
# Calculate the average
average_pin_tip = df['Pin (Tip)'].mean()
print(f"Final Answer: {average_pin_tip:.1f}")