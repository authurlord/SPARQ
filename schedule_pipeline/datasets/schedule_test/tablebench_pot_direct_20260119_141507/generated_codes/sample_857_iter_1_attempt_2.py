import pandas as pd

df = pd.read_csv('table.csv')

# Extract values for Model 8Ab and 8B
# Use exact column names as given in the table
displacement_8Ab = df.loc[df['Model'] == '8Ab', 'Displacement (l)'].values[0]
weight_8Ab = df.loc[df['Model'] == '8Ab', 'Weight(kg)'].values[0]

displacement_8B = df.loc[df['Model'] == '8B', 'Displacement (l)'].values[0]
weight_8B = df.loc[df['Model'] == '8B', 'Weight(kg)'].values[0]

# Calculate differences
diff_8Ab = abs(float(displacement_8Ab) - float(weight_8Ab))
diff_8B = abs(float(displacement_8B) - float(weight_8B))

# Compare and determine which has greater difference
if diff_8Ab > diff_8B:
    final_answer = "8Ab"
else:
    final_answer = "8B"

print(f"Final Answer: {final_answer}")