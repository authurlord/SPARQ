import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for Córdoba and Valencia
cordoba_row = df[df['Tribunal'] == 'Córdoba']
valencia_row = df[df['Tribunal'] == 'Valencia']

# Extract execution values, handling non-numeric entries
def safe_numeric(value):
    if isinstance(value, str):
        # Remove any question marks or non-numeric characters
        cleaned = value.replace('?', '').replace('?', '')
        if cleaned.strip():
            return int(cleaned)
    return 0

# Calculate total executions for Córdoba
cordoba_executions = safe_numeric(cordoba_row['Executions in persona'].iloc[0]) + safe_numeric(cordoba_row['Executions in effigie'].iloc[0])

# Calculate total executions for Valencia
valencia_executions = safe_numeric(valencia_row['Executions in persona'].iloc[0]) + safe_numeric(valencia_row['Executions in effigie'].iloc[0])

# Compute the difference
difference = cordoba_executions - valencia_executions

print(f"Final Answer: {difference}")