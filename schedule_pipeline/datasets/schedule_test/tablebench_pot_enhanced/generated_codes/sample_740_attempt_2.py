import pandas as pd

df = pd.read_csv('table.csv')

# Extract the last three periods for trend analysis
recent_data = df.tail(3)

# Calculate average annual change for CBR, CDR, IMR
cbg_change = (recent_data['CBR1'].iloc[-1] - recent_data['CBR1'].iloc[-2]) / 5
cdr_change = (recent_data['CDR1'].iloc[-1] - recent_data['CDR1'].iloc[-2]) / 5
imr_change = (recent_data['IMR1'].iloc[-1] - recent_data['IMR1'].iloc[-2]) / 5

# Project for 2010–2015 (next 5-year period)
projected_cbr = float(recent_data['CBR1'].iloc[-1]) + cbg_change
projected_cdr = float(recent_data['CDR1'].iloc[-1]) + cdr_change
projected_imr = float(recent_data['IMR1'].iloc[-1]) + imr_change

# Round to one decimal place
print(f"Final Answer: {projected_cbr:.1f}, {projected_cdr:.1f}, {projected_imr:.1f}")