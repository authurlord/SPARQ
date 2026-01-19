import pandas as pd

df = pd.read_csv('table.csv')
# Extract displacement and weight for 8Ab and 8B
displacement_8Ab = float(df.loc[df['Model'] == 'Displacement (l)', '8Ab'])
weight_8Ab = float(df.loc[df['Model'] == 'Weight(kg)', '8Ab'])
displacement_8B = float(df.loc[df['Model'] == 'Displacement (l)', '8B'])
weight_8B = float(df.loc[df['Model'] == 'Weight(kg)', '8B'])

# Calculate differences
diff_8Ab = abs(displacement_8Ab - weight_8Ab)
diff_8B = abs(displacement_8B - weight_8B)

# Determine which has greater difference
if diff_8Ab > diff_8B:
    result = "8Ab"
else:
    result = "8B"

print(f"Final Answer: {result}")