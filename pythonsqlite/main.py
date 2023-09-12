#https://www.askpython.com/python/examples/insert-multiple-records-sqlite3#:~:text=Inserting%20values%20into%20the%20database&text=The%20SQLite%20module's%20two%20methods,many%20records%20at%20a%20time.
#https://www.kaggle.com/code/jmcaro/machine-learning-classifiers-wheat-seeds
#https://www.geeksforgeeks.org/python-sqlite-deleting-data-in-table/?ref=lbp
#https://www.youtube.com/watch?v=M-4EpNdlSuY
#https://www.alixaprodev.com/how-to-get-column-names-from-sqlite-database-table-in-python/
#https://stackoverflow.com/questions/14431646/how-to-write-pandas-dataframe-to-sqlite-with-index
#https://datatofish.com/pandas-dataframe-to-sql/
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
#https://realpython.com/python-print/
#https://stackoverflow.com/questions/51422062/how-can-i-replace-int-values-with-string-values-in-a-dataframe

import sqlite3
import pandas as pd

seeds = pd.read_csv('/Users/mirella/Desktop/mestrado/Redes Neurais Não Supervisionadas/trab/seeds/seeds.txt', sep=r'\t',engine='python')
XX = seeds.iloc[:, 0:8].values
dXX = pd.DataFrame(XX, columns = ['Area', 'Perimeter','Compactness','Kernel_Length', 'Kernel_Width', 'Asymmetry_Coefficient','Kernel_Groove_Length', 'Original_Labels'])
dXX.loc[-1] = [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 1]  # adding a row
dXX.index = dXX.index + 1  # shifting index
dXX.sort_index(inplace=True)
print("Original Dataframe:", dXX)

#Map syntax
dXX['Original_Labels'] = dXX['Original_Labels'].map({1.0:'Kama', 2.0:'Rosa', 3.0:'Canadian'})

print("Modified Dataframe:", dXX)

seeeds = sqlite3.connect('wheat_seeds.db')

cursor = seeeds.cursor()

#Drop the seeds table if already exists
cursor.execute("DROP TABLE IF EXISTS WSD")

##Create new table from a pandas database. WSD means Wheat Seeds Dataset
#cursor.execute("CREATE TABLE WSD (Area float, Perimeter float, Compactness float, Kernel_Length float, Kernel__Width float, Asymmetry_Coefficient float, Kernel_Groove_Length float, Wheat_Varieties text)")
dXX.to_sql(name='WSD', con=seeeds, index=False)#index_label= ['Area', 'Perimeter','Compactness','Kernel Length', 'Kernel Width', 'Asymmetry Coefficient','Kernel Groove Length', 'Original Labels'])

##Adding one record at a time
#cursor.execute("INSERT INTO WSD(15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 'Kama')")
#cursor.execute("INSERT INTO WSD(14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956, 'Kama')")

##Adding multiple records at a time
#multiple_columns = [(15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 'Kama'),
#                    (14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956, 'Kama'),
#                    (14.29, 14.09, 0.905, 5.291, 3.337, 2.699, 4.825, 'Kama'),
#                    (13.84, 13.94, 0.8955, 5.324, 3.379, 2.259, 4.805, 'Kama')]
#cursor.executemany("INSERT INTO WSD VALUES (?, ?, ?, ?, ?, ?, ?, ?)", multiple_columns)

#Discover the rows number of the dataset
cursor.execute('SELECT * FROM WSD')
length = cursor.fetchall()
length = len(length)
print("number of rows:", length)

# #Select a specific column of the dataset
# cursor.execute("SELECT Perimeter FROM WSD")
# for row in cursor.fetchall():
#     for i in range(1,length):
#         print("Perimeter[" + str(i) + "]: " + str(row))

#Delete data
cursor.execute("SELECT Area FROM WSD")
cursor.execute("DELETE FROM WSD WHERE Area < 15.26")

##Delete all data from the table
#cursor_obj.execute("DELETE FROM WSD")

##Obtain the description of the sqlite database
#desc = cursor.description
#print(desc)

##Replace an item for another
#cursor.execute("UPDATE WSD SET Original_Labels = REPLACE(1, 1, 'Kama')")

##Drop a specific column
#cursor.execute("ALTER TABLE WSD DROP COLUMN Original_Labels")

##Obtain the name of the columns of the sqlite database
#names = [description[0] for description in cursor.description]
#print(names)

#Show the dataset
for row in cursor.execute('SELECT * FROM WSD'):
    print(row)

#Show the dataset summed up
#cursor.execute("SELECT * FROM WSD")
#print(cursor.fetchall())

seeeds.commit()

seeeds.close()