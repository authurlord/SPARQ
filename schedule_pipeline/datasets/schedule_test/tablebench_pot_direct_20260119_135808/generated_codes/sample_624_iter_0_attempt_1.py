import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Canada (average)
canada_row = df[df['government'] == 'canada (average)']
federal_tax = float(canada_row['federal excise tax ( cad / l )'].values[0])
total_tax = float(canada_row['total excise tax (cad / l)'].values[0])

# Increase federal excise tax by 20%
new_federal_tax = federal_tax * 1.2

# Recalculate total excise tax with increased federal tax
new_total_tax = new_federal_tax + (total_tax - federal_tax)

print(f"Final Answer: {new_total_tax:.2f}")