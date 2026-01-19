import pandas as pd

df = pd.read_csv('table.csv')
# Extract Conservative councillors and years
conservative_councillors = df['Conservative councillors']
years = df['Year'].astype(int)

# Calculate the total change from 1947 to 1972
start_year = 1947
end_year = 1972
change = conservative_councillors.iloc[-1] - conservative_councillors.iloc[0]
num_years = end_year - start_year + 1

# Average annual change
average_annual_change = change / num_years
print(f"Final Answer: {average_annual_change:.2f}")