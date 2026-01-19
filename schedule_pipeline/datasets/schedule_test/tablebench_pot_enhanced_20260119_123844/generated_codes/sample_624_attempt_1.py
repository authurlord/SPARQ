import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for 'canada (average)'
canada_row = df[df['government'] == 'canada (average)']
# Extract federal excise tax and total excise tax
old_federal = float(canada_row['federal excise tax ( cad / l )'].values[0])
old_total = float(canada_row['total excise tax (cad / l)'].values[0])
# Calculate provincial portion
provincial_portion = old_total - old_federal
# New federal excise tax after 20% increase
new_federal = old_federal * 1.20
# New total excise tax
new_total = new_federal + provincial_portion
print(f"Final Answer: {new_total:.1f}")