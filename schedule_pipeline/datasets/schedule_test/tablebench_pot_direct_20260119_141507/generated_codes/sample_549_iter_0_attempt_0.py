import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'district - wide' column
district_wide = df['district - wide'].astype(int)
# Calculate the average annual change
initial = district_wide.iloc[0]  # 2001-2002
final = district_wide.iloc[-1]  # 2006-2007
annual_change = (final - initial) / 5
print(f"Final Answer: {annual_change:.1f}")