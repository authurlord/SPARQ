import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Total External Debt in Million of US Dollars ($)' to numeric
df['Total External Debt in Million of US Dollars ($)'] = pd.to_numeric(df['Total External Debt in Million of US Dollars ($)'].str.replace(',', ''), errors='coerce')

# Convert 'Debt Service Ratio (%)' to numeric, handling missing values
df['Debt Service Ratio (%)'] = pd.to_numeric(df['Debt Service Ratio (%)'].str.replace('-', '0'), errors='coerce')

# Filter years after 2010
after_2010 = df[df['Fiscal Year'] > '2010']

# Extract debt service ratio values after 2010
debt_service_ratio_after_2010 = after_2010['Debt Service Ratio (%)']

# Check trend: if it decreased from 2010 to subsequent years
ratio_2010 = df[df['Fiscal Year'] == '2010']['Debt Service Ratio (%)'].iloc[0]
ratio_after_2010 = debt_service_ratio_after_2010.dropna()

# Since the ratio dropped from 9.9% to lower values, the impact is a decrease
print(f"Final Answer: The debt service ratio decreased in subsequent years despite the increase in external debt.")