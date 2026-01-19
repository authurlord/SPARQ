import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average of the 'Pin (Tip)' column
average_pin_tip = df['Pin (Tip)'].mean()
print(f"Final Answer: {average_pin_tip:.1f}")