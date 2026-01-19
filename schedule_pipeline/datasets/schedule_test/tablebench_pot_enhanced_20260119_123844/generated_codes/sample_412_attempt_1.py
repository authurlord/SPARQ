import pandas as pd

df = pd.read_csv('table.csv')

# Display the first few rows to understand the data
print("Table Structure:")
print(df.head())

# Describe the key columns
print("\nKey Columns and Their Features:")
print("- year: Represents the year or range of years (e.g., 1913, 191822).")
print("- typhus: Number of typhus cases reported.")
print("- typhoid fever: Number of typhoid fever cases reported.")
print("- relapsing fever: Number of relapsing fever cases reported.")
print("- smallpox: Number of smallpox cases reported.")
print("- malaria: Number of malaria cases reported, with some entries having averages (e.g., '2940 (avg)')")

# Final Answer: Summarize the main features
print(f"Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria")