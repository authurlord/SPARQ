import pandas as pd

df = pd.read_csv('table.csv')
# Extract the R class in service at the start of 1931 and the quantity withdrawn
start_r_class = df[df['Year'] == '1931']['R class in service at start of year'].iloc[0]
withdrawn = df[df['Year'] == '1931']['Quantity withdrawn'].iloc[0]

# Calculate the number of R class in service at the end of 1931
end_r_class = int(start_r_class) - int(withdrawn)
print(f"Final Answer: {end_r_class}")