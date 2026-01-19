import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Canada (average)
canada_row = df[df['government'] == 'canada (average)']
# Extract the current total excise tax
current_total_tax = float(canada_row['total excise tax (cad / l)'].values[0])
# Calculate new total excise tax after 20% increase
new_total_tax = current_total_tax * 1.20
print(f"Final Answer: {new_total_tax:.2f}")