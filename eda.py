import sqlite3
import pandas as pd
from prettytable import PrettyTable

# Load CSV into pandas DataFrame
df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv"
)

with sqlite3.connect("my_data1.db") as con:
    cur = con.cursor()
    
    # Write DataFrame to SQLite table
    df.to_sql("SPACEXTBL", con, if_exists='replace', index=False)
    
    # ----------------------
    # Task 6: Boosters with successful drone ship landing and payload mass between 4000-6000
    # ----------------------
    cur.execute('''
        SELECT "BoosterVersion", "PayloadMass"
        FROM SPACEXTBL
        WHERE "Outcome"="True ASDS" AND "PayloadMass" > 4000 AND "PayloadMass" < 6000;
    ''')
    rows_task6 = cur.fetchall()
    table6 = PrettyTable()
    table6.field_names = ["BoosterVersion", "PayloadMass"]
    for row in rows_task6:
        table6.add_row(row)
    print("Task 6: Boosters with successful drone ship landing and payload mass 4000-6000")
    print(table6)
    
    # ----------------------
    # Task 7: Total number of successful and failure mission outcomes
    # ----------------------
    cur.execute('''
        SELECT "Outcome", COUNT(*) 
        FROM SPACEXTBL
        GROUP BY "Outcome";
    ''')
    rows_task7 = cur.fetchall()
    table7 = PrettyTable()
    table7.field_names = ["Outcome", "Count"]
    for row in rows_task7:
        table7.add_row(row)
    print("\nTask 7: Total number of successful and failed mission outcomes")
    print(table7)
    
    # ----------------------
    # Task 8: Booster versions that carried the maximum payload mass
    # ----------------------
    cur.execute('''
        SELECT "BoosterVersion", "PayloadMass"
        FROM SPACEXTBL
        WHERE "PayloadMass" = (SELECT MAX("PayloadMass") FROM SPACEXTBL);
    ''')
    rows_task8 = cur.fetchall()
    table8 = PrettyTable()
    table8.field_names = ["BoosterVersion", "PayloadMass"]
    for row in rows_task8:
        table8.add_row(row)
    print("\nTask 8: Booster versions that carried the maximum payload mass")
    print(table8)
    
    # ----------------------
    # Task 9: Records with month names, failure drone ship outcomes, booster versions, launch_site for year 2015
    # ----------------------
    cur.execute('''
        SELECT 
            substr("Date",6,2) AS Month, 
            "Outcome", 
            "BoosterVersion", 
            "LaunchSite"
        FROM SPACEXTBL
        WHERE substr("Date",1,4)='2015' AND "Outcome" LIKE 'False ASDS';
    ''')
    rows_task9 = cur.fetchall()
    table9 = PrettyTable()
    table9.field_names = ["Month", "Outcome", "BoosterVersion", "LaunchSite"]
    for row in rows_task9:
        table9.add_row(row)
    print("\nTask 9: Failure drone ship landings in 2015 by month")
    print(table9)
    
    # ----------------------
    # Task 10: Rank count of landing outcomes between 2010-06-04 and 2017-03-20
    # ----------------------
    cur.execute('''
        SELECT "Outcome", COUNT(*) AS OutcomeCount
        FROM SPACEXTBL
        WHERE "Date" >= '2010-06-04' AND "Date" <= '2017-03-20'
        GROUP BY "Outcome"
        ORDER BY OutcomeCount DESC;
    ''')
    rows_task10 = cur.fetchall()
    table10 = PrettyTable()
    table10.field_names = ["Outcome", "Count"]
    for row in rows_task10:
        table10.add_row(row)
    print("\nTask 10: Ranked count of landing outcomes between 2010-06-04 and 2017-03-20")
    print(table10)
