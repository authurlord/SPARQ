import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display a detailed description of the table
print("Table Description:")
print("Columns:")
print(" - Fiscal Year: The fiscal year for which the data is reported.")
print(" - Total External Debt in Million of US Dollars ($): Total debt owed to foreign creditors in millions of USD.")
print(" - Total Debt Service in Million of US Dollars ($): Total amount paid annually for interest and principal repayment.")
print(" - External Debt to GDP Ratio (%): Ratio of external debt to GDP, indicating the debt burden relative to economic size.")
print(" - Debt Service Ratio (%): Ratio of debt service to GDP, reflecting the proportion of GDP used to repay debt.")

print("\nBasic Insights:")
print(" - Total external debt peaked in 2012 at $79,949 million and has since stabilized.")
print(" - External debt to GDP ratio declined from 68.2% (2001) to 23.5% (2018), indicating improved debt sustainability.")
print(" - Debt service ratio dropped from 17.1% (2002) to below 10% (from 2008 onward), suggesting reduced financial pressure.")
print(" - Debt service ratio is marked as '-' from 2015 onwards, indicating missing or unavailable data for those years.")