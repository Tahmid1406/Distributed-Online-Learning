import mysql.connector

myDB = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="1115271831",

)

my_cursor = myDB.cursor()

my_cursor.execute("CREATE DATABASE crypto")

my_cursor.execute("SHOW DATABASES")

for db in my_cursor:
    print(db)