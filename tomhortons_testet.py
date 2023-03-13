import sqlite3
import pandas as pd
from haversine import haversine

north_pole = (90,0)
weight_limit = 1000.0

def bb_sort(ll):
    s_limit = 5000
    optimal = False
    ll = [[0,north_pole,10]] + ll[:] + [[0,north_pole,10]] 
    while not optimal:
        optimal = True
        for i in range(1,len(ll) - 2):
            lcopy = ll[:]
            lcopy[i], lcopy[i+1] = ll[i+1][:], ll[i][:]
            if path_opt_test(ll[1:-1]) > path_opt_test(lcopy[1:-1]):
                #print("swap")
                ll = lcopy[:]
                optimal = False
                s_limit -= 1
                if s_limit < 0:
                    optimal = True
                    break
    return ll[1:-1]

def path_opt_test(llo):
    f_ = 0.0
    d_ = 0.0
    l_ = north_pole
    for i in range(len(llo)):
        d_ += haversine(l_, llo[i][1])
        f_ += d_ * llo[i][2]
        l_ = llo[i][1]
    d_ += haversine(l_, north_pole)
    f_ += d_ * 10 #sleigh weight for whole trip
    return f_

gifts = pd.read_csv("../input/gifts.csv").fillna(" ")
c = sqlite3.connect(":memory:")
gifts.to_sql("gifts",c)
cu = c.cursor()
cu.execute("ALTER TABLE gifts ADD COLUMN 'TripId' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'i' INT;")
cu.execute("ALTER TABLE gifts ADD COLUMN 'j' INT;")
c.commit()


for n in [1.25252525]:
    i_ = 0
    j_ = 0
    for i in range(90,-90,int(-180/n)):
        i_ += 1
        j_ = 0
        for j in range(180,-180,int(-360/n)):
            j_ += 1
            cu = c.cursor()
            cu.execute("UPDATE gifts SET i=" + str(i_) + ", j=" + str(j_) + " WHERE ((Latitude BETWEEN " + str(i - (180/n)) + " AND  " + str(i) + ") AND (Longitude BETWEEN " + str(j - (360/n)) + " AND  " + str(j) + "));")
            c.commit()
    
    for limit_ in [67]:
        trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY i, j, Longitude, Latitude LIMIT " + str(limit_) + " ) ORDER BY Latitude DESC",c)
        t_ = 0
        while len(trips.GiftId)>0:
            g = []
            t_ += 1
            w_ = 0.0
            for i in range(len(trips.GiftId)):
                if (w_ + float(trips.Weight[i]))<= weight_limit:
                    w_ += float(trips.Weight[i])
                    g.append(trips.GiftId[i])
            cu = c.cursor()
            cu.execute("UPDATE gifts SET TripId = " + str(t_) + " WHERE GiftId IN(" + (",").join(map(str,g)) + ");")
            c.commit()
        
            trips = pd.read_sql("SELECT * FROM (SELECT * FROM gifts WHERE TripId IS NULL ORDER BY i, j, Longitude, Latitude LIMIT " + str(limit_) + " ) ORDER BY Latitude DESC",c)
            #break
        
        ou_ = open("submission_opt" + str(limit_) + " " + str(n) + ".csv","w")