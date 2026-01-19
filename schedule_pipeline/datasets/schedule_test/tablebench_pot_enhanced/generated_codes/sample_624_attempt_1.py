import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'canada (average)'
canada_row = df[df['government'] == 'canada (average)']
federal_tax = float(canada_row['federal excise tax ( cad / l )'].values[0])
total_tax = float(canada_row['total excise tax (cad / l)'].values[0])

# Calculate the increased federal tax
increased_federal_tax = federal_tax * 1.20

# The provincial tax is the difference between total and federal tax
provincial_tax = total_tax - federal_tax

# New total excise tax
new_total_tax = increased_federal_tax + provincial_tax

print(f"Final Answer: {new_total_tax:.1f}")