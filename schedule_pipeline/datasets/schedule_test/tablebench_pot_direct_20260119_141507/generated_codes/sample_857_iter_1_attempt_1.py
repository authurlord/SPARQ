import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Identify the row index for "Displacement (l)"
displacement_row = df[df['Model'] == 'Displacement (l)'].index[0]

# Extract displacement and weight values for 8Ab and 8B
displacement_8Ab = df.loc[displacement_row, '8Aa']  # Correctly access by column name
displacement_8B = df.loc[displacement_row, '8B']

weight_8Ab = df.loc[displacement_row + 1, '8Aa']  # Weight row is next after displacement
weight_8B = df.loc[displacement_row + 1, '8B']

# Compute differences
diff_8Ab = float(displacement_8Ab) - float(weight_8Ab)
diff_8B = float(displacement_8B) - float(weight_8B)

# Compare and find which model has a greater difference
if diff_8Ab > diff_8B:
    final_model = '8Ab'
else:
    final_model = '8B'

print(f"Final Answer: {final_model}")