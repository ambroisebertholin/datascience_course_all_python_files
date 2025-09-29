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
    
    # # ----------------------
    # # Task 6: Boosters with successful drone ship landing and payload mass between 4000-6000
    # # ----------------------
    # cur.execute('''
    #     SELECT "BoosterVersion", "PayloadMass"
    #     FROM SPACEXTBL
    #     WHERE "Outcome"="True ASDS" AND "PayloadMass" > 4000 AND "PayloadMass" < 6000;
    # ''')
    # rows_task6 = cur.fetchall()
    # table6 = PrettyTable()
    # table6.field_names = ["BoosterVersion", "PayloadMass"]
    # for row in rows_task6:
    #     table6.add_row(row)
    # print("Task 6: Boosters with successful drone ship landing and payload mass 4000-6000")
    # print(table6)
    
    # # ----------------------
    # # Task 7: Total number of successful and failure mission outcomes
    # # ----------------------
    # cur.execute('''
    #     SELECT "Outcome", COUNT(*) 
    #     FROM SPACEXTBL
    #     GROUP BY "Outcome";
    # ''')
    # rows_task7 = cur.fetchall()
    # table7 = PrettyTable()
    # table7.field_names = ["Outcome", "Count"]
    # for row in rows_task7:
    #     table7.add_row(row)
    # print("\nTask 7: Total number of successful and failed mission outcomes")
    # print(table7)
    
    # # ----------------------
    # # Task 8: Booster versions that carried the maximum payload mass
    # # ----------------------
    # cur.execute('''
    #     SELECT "BoosterVersion", "PayloadMass"
    #     FROM SPACEXTBL
    #     WHERE "PayloadMass" = (SELECT MAX("PayloadMass") FROM SPACEXTBL);
    # ''')
    # rows_task8 = cur.fetchall()
    # table8 = PrettyTable()
    # table8.field_names = ["BoosterVersion", "PayloadMass"]
    # for row in rows_task8:
    #     table8.add_row(row)
    # print("\nTask 8: Booster versions that carried the maximum payload mass")
    # print(table8)
    
    # # ----------------------
    # # Task 9: Records with month names, failure drone ship outcomes, booster versions, launch_site for year 2015
    # # ----------------------
    # cur.execute('''
    #     SELECT 
    #         substr("Date",6,2) AS Month, 
    #         "Outcome", 
    #         "BoosterVersion", 
    #         "LaunchSite"
    #     FROM SPACEXTBL
    #     WHERE substr("Date",1,4)='2015' AND "Outcome" LIKE 'False ASDS';
    # ''')
    # rows_task9 = cur.fetchall()
    # table9 = PrettyTable()
    # table9.field_names = ["Month", "Outcome", "BoosterVersion", "LaunchSite"]
    # for row in rows_task9:
    #     table9.add_row(row)
    # print("\nTask 9: Failure drone ship landings in 2015 by month")
    # print(table9)
    
    # # ----------------------
    # # Task 10: Rank count of landing outcomes between 2010-06-04 and 2017-03-20
    # # ----------------------
    # cur.execute('''
    #     SELECT "Outcome", COUNT(*) AS OutcomeCount
    #     FROM SPACEXTBL
    #     WHERE "Date" >= '2010-06-04' AND "Date" <= '2017-03-20'
    #     GROUP BY "Outcome"
    #     ORDER BY OutcomeCount DESC;
    # ''')
    # rows_task10 = cur.fetchall()
    # table10 = PrettyTable()
    # table10.field_names = ["Outcome", "Count"]
    # for row in rows_task10:
    #     table10.add_row(row)
    # print("\nTask 10: Ranked count of landing outcomes between 2010-06-04 and 2017-03-20")
    # print(table10)


    # =====================
    # NEW QUERIES FOR DOCUMENT
    # =====================
    
    # ----------------------
    # Unique Launch Site Names
    # ----------------------
    print("\n" + "=" * 70)
    print("UNIQUE LAUNCH SITES")
    print("=" * 70)
    cur.execute('''
        SELECT DISTINCT "Launch_Site"
        FROM SPACEXTBL
        ORDER BY "Launch_Site";
    ''')
    rows_launchsites = cur.fetchall()
    table_launchsites = PrettyTable()
    table_launchsites.field_names = ["Launch_Site"]
    for row in rows_launchsites:
        table_launchsites.add_row(row)
    print(table_launchsites)
    
    # ----------------------
    # Launch Sites Beginning with 'CCA'
    # ----------------------
    print("\n" + "=" * 70)
    print("LAUNCH SITES BEGINNING WITH 'CCA'")
    print("=" * 70)
    cur.execute('''
        SELECT "Launch_Site"
        FROM SPACEXTBL
        WHERE "Launch_Site" LIKE 'CCA%'
        LIMIT 5;
    ''')
    rows_cca = cur.fetchall()
    table_cca = PrettyTable()
    table_cca.field_names = ["Launch_Site"]
    for row in rows_cca:
        table_cca.add_row(row)
    print(table_cca)
    
    # ----------------------
    # Total Payload Mass Carried by Boosters
    # ----------------------
    print("\n" + "=" * 70)
    print("TOTAL PAYLOAD MASS")
    print("=" * 70)
    cur.execute('''
        SELECT SUM("PAYLOAD_MASS__KG_") as TotalPayloadMass
        FROM SPACEXTBL;
    ''')
    total_payload = cur.fetchone()[0]
    print(f"Total Payload Mass: {total_payload or 0:,.0f} kg")
    
    # ----------------------
    # Average Payload Mass for F9 v1.1
    # ----------------------
    print("\n" + "=" * 70)
    print("AVERAGE PAYLOAD MASS FOR FALCON 9 v1.1")
    print("=" * 70)
    cur.execute('''
        SELECT AVG("PAYLOAD_MASS__KG_") as AvgPayloadMass
        FROM SPACEXTBL
        WHERE "Booster_Version" LIKE '%v1.1%';
    ''')
    avg_payload = cur.fetchone()[0]
    print(f"Average Payload Mass for F9 v1.1: {avg_payload or 0:,.0f} kg")
    
    # ----------------------
    # First Successful Landing
    # ----------------------
    print("\n" + "=" * 70)
    print("FIRST SUCCESSFUL LANDING")
    print("=" * 70)
    cur.execute('''
        SELECT "Date", "Booster_Version", "Launch_Site", "Landing_Outcome"
        FROM SPACEXTBL
        WHERE "Landing_Outcome" LIKE '%Success%' OR "Landing_Outcome" LIKE '%True%'
        ORDER BY "Date"
        LIMIT 1;
    ''')
    first_success = cur.fetchone()
    if first_success:
        table_success = PrettyTable()
        table_success.field_names = ["Date", "Booster_Version", "Launch_Site", "Landing_Outcome"]
        table_success.add_row(first_success)
        print(table_success)
    else:
        print("No successful landings found")
    
        # =====================
    # ADDITIONAL QUERIES
    # =====================
    
    # ----------------------
    # Boosters with successful drone ship landing and payload 4000-6000 kg
    # ----------------------
    print("\n" + "=" * 70)
    print("SUCCESSFUL DRONE SHIP LANDINGS (4000-6000 kg)")
    print("=" * 70)
    cur.execute('''
        SELECT "Booster_Version", "PAYLOAD_MASS__KG_", "Landing_Outcome"
        FROM SPACEXTBL
        WHERE ("Landing_Outcome" LIKE '%True%' OR "Landing_Outcome" LIKE '%Success%')
        AND "PAYLOAD_MASS__KG_" > 4000 AND "PAYLOAD_MASS__KG_" < 6000;
    ''')
    rows_drone = cur.fetchall()
    table_drone = PrettyTable()
    table_drone.field_names = ["Booster_Version", "Payload_Mass_kg", "Landing_Outcome"]
    for row in rows_drone:
        table_drone.add_row(row)
    print(table_drone)
    
    # ----------------------
    # Total number of successful and failure mission outcomes
    # ----------------------
    print("\n" + "=" * 70)
    print("MISSION OUTCOMES SUMMARY")
    print("=" * 70)
    cur.execute('''
        SELECT "Mission_Outcome", COUNT(*) 
        FROM SPACEXTBL
        GROUP BY "Mission_Outcome";
    ''')
    rows_mission = cur.fetchall()
    table_mission = PrettyTable()
    table_mission.field_names = ["Mission_Outcome", "Count"]
    for row in rows_mission:
        table_mission.add_row(row)
    print(table_mission)
    
    # ----------------------
    # Boosters that carried maximum payload mass
    # ----------------------
    print("\n" + "=" * 70)
    print("MAXIMUM PAYLOAD MASS CARRIERS")
    print("=" * 70)
    cur.execute('''
        SELECT "Booster_Version", "PAYLOAD_MASS__KG_"
        FROM SPACEXTBL
        WHERE "PAYLOAD_MASS__KG_" = (SELECT MAX("PAYLOAD_MASS__KG_") FROM SPACEXTBL);
    ''')
    rows_max_payload = cur.fetchall()
    table_max = PrettyTable()
    table_max.field_names = ["Booster_Version", "Payload_Mass_kg"]
    for row in rows_max_payload:
        table_max.add_row(row)
    print(table_max)
    
    # ----------------------
    # Failed drone ship landings in 2015
    # ----------------------
    print("\n" + "=" * 70)
    print("FAILED DRONE SHIP LANDINGS IN 2015")
    print("=" * 70)
    cur.execute('''
        SELECT 
            substr("Date",1,7) AS Month,
            "Booster_Version", 
            "Launch_Site",
            "Landing_Outcome"
        FROM SPACEXTBL
        WHERE substr("Date",1,4) = '2015' 
        AND ("Landing_Outcome" LIKE '%False%' OR "Landing_Outcome" LIKE '%Failure%');
    ''')
    rows_2015_failures = cur.fetchall()
    table_2015 = PrettyTable()
    table_2015.field_names = ["Month", "Booster_Version", "Launch_Site", "Landing_Outcome"]
    for row in rows_2015_failures:
        table_2015.add_row(row)
    print(table_2015)
    
    # ----------------------
    # Rank landing outcomes between 2010-06-04 and 2017-03-20
    # ----------------------
    print("\n" + "=" * 70)
    print("RANKED LANDING OUTCOMES (2010-06-04 to 2017-03-20)")
    print("=" * 70)
    cur.execute('''
        SELECT "Landing_Outcome", COUNT(*) AS OutcomeCount
        FROM SPACEXTBL
        WHERE "Date" >= '2010-06-04' AND "Date" <= '2017-03-20'
        GROUP BY "Landing_Outcome"
        ORDER BY OutcomeCount DESC;
    ''')
    rows_ranked = cur.fetchall()
    table_ranked = PrettyTable()
    table_ranked.field_names = ["Landing_Outcome", "Count"]
    for row in rows_ranked:
        table_ranked.add_row(row)
    print(table_ranked)
