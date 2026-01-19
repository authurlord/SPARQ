import pandas as pd

df = pd.read_csv('table.csv')

# Extract the last two periods to calculate average annual decline
last_period = df.iloc[-1]  # 2005-2010
second_last = df.iloc[-2]  # 2000-2005

# Calculate average annual decline
cbra = (float(second_last['CBR1']) - float(last_period['CBR1'])) / 5
cdra = (float(second_last['CDR1']) - float(last_period['CDR1'])) / 5
imra = (float(second_last['IMR1']) - float(last_period['IMR1'])) / 5

# Project for 2010-2015 (5 years ahead)
projected_cbr = float(last_period['CBR1']) - (cbra * 5)
projected_cdr = float(last_period['CDR1']) - (cdra * 5)
projected_imr = float(last_period['IMR1']) - (imra * 5)

print(f"Final Answer: {projected_cbr:.1f}, {projected_cdr:.1f}, {projected_imr:.1f}")