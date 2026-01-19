import pandas as pd

df = pd.read_csv('table.csv')
# Display the column names and a brief description
print("Key columns in the table:")
print("year: Represents the year or range of years for the data.")
print("typhus: Number of typhus cases reported.")
print("typhoid fever: Number of typhoid fever cases reported.")
print("relapsing fever: Number of relapsing fever cases reported.")
print("smallpox: Number of smallpox cases reported.")
print("malaria: Number of malaria cases reported, with 'avg' indicating an average value in one row.")

# Final answer is a summary of the main features
print(f"Final Answer: year, typhus, typhoid fever, relapsing fever, smallpox, malaria")