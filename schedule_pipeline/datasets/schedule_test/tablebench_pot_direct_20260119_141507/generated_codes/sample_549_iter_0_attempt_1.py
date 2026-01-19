import pandas as pd

df = pd.read_csv('table.csv')
# Extract 'district - wide' values
district_wide = df['district - wide'].astype(int)
# Calculate the change from 2001-2002 to 2006-2007
initial = district_wide.iloc[0]  # 2001-2002
final = district_wide.iloc[5]    # 2006-2007
total_change = final - initial
average_annual_change = total_change / 5
print(f"Final Answer: {average_annual_change:.1f}")