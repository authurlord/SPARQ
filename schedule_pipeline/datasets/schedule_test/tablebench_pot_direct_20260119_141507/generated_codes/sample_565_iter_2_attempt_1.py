- Row 0: Year=1820, Year_1=1885 → not matching
   But note: the increase from 1850 to 1855: 200,877 - 369,980 = negative
   1855 to 1860: 153,640 - 200,877 = negative
   1860 to 1865: 248,120 - 153,640 = positive (94,480)
   1865 to 1870: 387,203 - 248,120 = positive (139,083)
   - 1855 → 1860: 153,640 - 200,877 = -47,237
   - 1860 → 1865: 248,120 - 153,640 = 94,480
   - 1865 → 1870: 387,203 - 248,120 = 139,083
import pandas as pd
df = pd.read_csv('table.csv')
years_of_interest = [1850, 1855, 1860, 1865, 1870]
values = df[df['Year'].isin(years_of_interest)]
values_list = []
    year = row['Year']
    value_str = row['-']  # second column
    value = int(value_str.replace(',', ''))
years_values = [(1850, 369980), (1855, 200877), (1860, 153640), (1865, 248120), (1870, 387203)]
differences = []