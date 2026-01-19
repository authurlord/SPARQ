import pandas as pd

df = pd.read_csv('table.csv')

# Display the table and describe each column
print("Table Description:")
print("Columns:")
print("- Applications: Number of applications received each year.")
print("- Offer Rate (%): Percentage of applicants who received an offer.")
print("- Enrols: Number of students who enrolled after receiving an offer.")
print("- Yield (%): Percentage of offer recipients who actually enrolled.")
print("- Applicant/Enrolled Ratio: Ratio of total applicants to enrollees; higher ratio indicates more competitive admissions.")
print("- Average Entry Tariff: Average cost of entry (e.g., tuition); 'n/a' indicates missing data.")

print("\nNotable Trends:")
print("- Applications increased steadily from 33,825 in 2013 to 41,000 in 2017.")
print("- Offer Rate remained high (86.4%–90.8%), with a slight dip in 2017.")
print("- Enrolments rose from 5,505 in 2013 to 6,065 in 2017.")
print("- Yield declined from 18.7% in 2013 to 16.5% in 2017, indicating lower acceptance rates among offer holders.")
print("- Applicant/Enrolled Ratio increased from 6.14 to 6.76, showing rising competition.")
print("- Average Entry Tariff dropped significantly in 2016 (to 176), then rose to 471 in 2015 (note: likely a data inconsistency), and was not available in 2017.")

# Final answer is a summary of key insights
print(f"Final Answer: Applications increased, Offer Rate stable, Enrols rose, Yield declined, Ratio increased, Tariff data inconsistent")