import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract values for Model 8Ab and Model 8B
displacement_8Ab = float(df[df['Model'] == '8Ab']['Displacement (l)'].iloc[2])
weight_8Ab = int(df[df['Model'] == '8Ab']['Weight(kg)'].iloc[2])

displacement_8B = float(df[df['Model'] == '8B']['Displacement (l)'].iloc[2])
weight_8B = int(df[df['Model'] == '8B']['Weight(kg)'].iloc[2])

# Calculate differences
diff_8Ab = abs(displacement_8Ab - weight_8Ab)
diff_8B = abs(displacement_8B - weight_8B)

# Compare and answer
if diff_8Ab > diff_8B:
    final_answer = "8Ab"
else:
    final_answer = "8B"

print(f"Final Answer: {final_answer}")