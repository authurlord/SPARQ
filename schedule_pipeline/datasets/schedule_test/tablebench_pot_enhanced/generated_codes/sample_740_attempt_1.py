import pandas as pd

df = pd.read_csv('table.csv')

# Extract the last two periods' values
last_period = df.iloc[-1]  # 2005-2010
second_last = df.iloc[-2]  # 2000-2005

# Calculate average annual change
cb_change = (last_period['CBR1'] - second_last['CBR1']) / 1
cd_change = (last_period['CDR1'] - second_last['CDR1']) / 1
imr_change = (last_period['IMR1'] - second_last['IMR1']) / 1

# Project next period (2010-2015)
projected_cbr = last_period['CBR1'] + cb_change
projected_cdr = last_period['CDR1'] + cd_change
projected_imr = last_period['IMR1'] + imr_change

print(f"Final Answer: {projected_cbr:.1f}, {projected_cdr:.1f}, {projected_imr:.1f}")