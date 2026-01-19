import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'introduced' column to integer for comparison
df['introduced'] = pd.to_numeric(df['introduced'], errors='coerce')

# Filter aircraft introduced before or in 2004 (fleet in 2004)
before_or_2004 = df[df['introduced'] <= 2004]
total_seating_2004 = before_or_2004['seating'].sum()

# Filter aircraft introduced after 2004 (fleet in 2008)
after_2004 = df[df['introduced'] > 2004]
total_seating_2008 = after_2004['seating'].sum()

# Change in total seating capacity from 2004 to 2008
change = total_seating_2008 - total_seating_2004

print(f"Final Answer: {change}")