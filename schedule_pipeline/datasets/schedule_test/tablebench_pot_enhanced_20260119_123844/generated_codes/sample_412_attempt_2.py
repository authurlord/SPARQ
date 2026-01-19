import pandas as pd

df = pd.read_csv('table.csv')
# Display the column names and their descriptions
print("Key columns in the table:")
print("- year: The year or year range of the data.")
print("- typhus: Number of typhus cases.")
print("- typhoid fever: Number of typhoid fever cases.")
print("- relapsing fever: Number of relapsing fever cases.")
print("- smallpox: Number of smallpox cases.")
print("- malaria: Number of malaria cases, with some entries showing averages (e.g., '2940 (avg)')")

# Final answer is a summary of the key columns
print("Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria")