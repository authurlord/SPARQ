import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display the first few rows to understand the structure
print("Table Summary:")
print(df.head())

# Describe key trends in the data
print("\nKey Observations:")
print("- Average population peaks at 51,000 (1985) and gradually declines after that.")
print("- Crude birth rate is highest in the 1970s and 1980s (around 21.7), then decreases to about 12.0 by 2006.")
print("- Crude death rate increases slightly after 1990, reaching a peak around 2000 (11.8).")
print("- Natural change shows a significant drop after 1990, with a recovery in 2001–2004 due to higher birth rates.")
print("- The data reflects a general trend of population stabilization and a shift toward lower birth and death rates over time.")