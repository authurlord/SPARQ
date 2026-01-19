import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to float
df['DC'] = pd.to_numeric(df['DC'], errors='coerce')
df['PSI'] = pd.to_numeric(df['PSI'], errors='coerce')
df['PCI'] = pd.to_numeric(df['PCI'], errors='coerce')

# Filter provinces with PCI > 12
filtered_df = df[df['PCI'] > 12]

# If no rows, handle appropriately
if filtered_df.empty:
    print("Final Answer: No provinces have PCI above 12")
else:
    # Calculate correlation between DC and PSI
    correlation = filtered_df['DC'].corr(filtered_df['PSI'])
    print(f"Final Answer: {correlation:.3f}")