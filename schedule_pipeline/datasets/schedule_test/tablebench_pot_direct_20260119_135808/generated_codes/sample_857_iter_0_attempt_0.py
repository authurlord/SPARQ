import pandas as pd

df = pd.read_csv('table.csv')
# Extract data for Model 8Ab and 8B
displacement_8Ab = float(df.loc[df['Model'] == 'Displacement (l)', '8Ab'].values[0])
weight_8Ab = int(df.loc[df['Model'] == 'Weight(kg)', '8Ab'].values[0])
difference_8Ab = abs(displacement_8Ab - weight_8Ab)

displacement_8B = float(df.loc[df['Model'] == 'Displacement (l)', '8B'].values[0])
weight_8B = int(df.loc[df['Model'] == 'Weight(kg)', '8B'].values[0])
difference_8B = abs(displacement_8B - weight_8B)

# Compare differences
if difference_8Ab > difference_8B:
    result = "8Ab"
else:
    result = "8B"

print(f"Final Answer: {result}")