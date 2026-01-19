import pandas as pd

df = pd.read_csv('table.csv')

# Display the first few rows to understand the data
print("Table Overview:")
print(df.head())

# Describe the key columns
print("\nKey Columns and Their Features:")
print("- year: Represents the year or range of years (e.g., '1913', '191822').")
print("- typhus: Number of typhus cases, ranging from 40 to 1300.")
print("- typhoid fever: Number of typhoid fever cases, ranging from 140 to 424.")
print("- relapsing fever: Cases ranging from 4 to 639.")
print("- smallpox: Cases ranging from 4 to 106.")
print("- malaria: Cases ranging from 2700 to 9924, with an average in 191822.")

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Final answer: List the key columns as per the question
print("Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria")