import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'district - wide' column and convert to numeric
district_wide = df['district - wide'].astype(float)

# Calculate the total change from 2001-2002 to 2006-2007
initial = district_wide.iloc[0]  # 2001-2002
final = district_wide.iloc[-1]   # 2006-2007
total_change = final - initial
num_years = 5  # from 2001-2002 to 2006-2007 is 5 years of change

average_annual_change = total_change / num_years
print(f"Final Answer: {average_annual_change:.1f}")