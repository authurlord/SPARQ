import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Display column descriptions
print("Column Descriptions:")
print("- Fiscal Year: The year for which the financial data is reported.")
print("- Total External Debt in Million of US Dollars ($): Total amount of external debt in millions of USD.")
print("- Total Debt Service in Million of US Dollars ($): Total amount paid annually for servicing external debt (interest and principal).")
print("- External Debt to GDP Ratio (%): Ratio of external debt to the country's GDP, indicating debt burden relative to economic size.")
print("- Debt Service Ratio (%): Ratio of total debt service to GDP, indicating the share of GDP spent on servicing debt.")

# Basic insights
print("\nBasic Insights:")
print(f"Total External Debt peaked in 2002 at ${df['Total External Debt in Million of US Dollars ($)'].max():,.0f} million and declined after 2012.")
print(f"External Debt to GDP Ratio peaked at {df['External Debt to GDP Ratio (%)'].max()}% in 2001 and fell to {df['External Debt to GDP Ratio (%)'].min()}% in 2018.")
print(f"Debt Service Ratio peaked at {df['Debt Service Ratio (%)'].max()}% in 2002 and decreased significantly after 2006, with values dropping below 13% from 2007 onward.")
print("Note: Debt Service Ratio is marked as '-' from 2015 onwards, suggesting data might be unavailable or not applicable for those years.")

Final Answer: Fiscal Year, Total External Debt, Total Debt Service, External Debt to GDP Ratio, Debt Service Ratio