import json
import random
import sys
from os import path

def p(jsonfile):
    with open(jsonfile) as d:
        x = 0
        for line in d:
            x += 1
            if x % 100000 == 0:
                print(x)
        print(x)

def process(json_file, prob=0.05):
    politics = ["anarchism", "anarcho_capitalism", "antiwork", "breadtube", "chapotraphouse", "communism", "completeanarchy", "conservative", "cringeanarchy", "democraticsocialism", "esist", "fullcommunism", "goldandblack", "jordanpeterson", "keep_track", "latestagecapitalism", "latestagesocialism", "liberal", "libertarian", "neoliberal", "onguardforthee", "ourpresident", "political_revolution", "politicalhumor", "politics", "progressive", "republican", "sandersforpresident", "selfawarewolves", "socialism", "the_donald", "the_mueller", "thenewright", "voteblue", "wayofthebern", "yangforpresidenthq"]
    sports = ["baseball", "boxing", "cricket", "football", "golf", "hockey", "mma", "nba", "nfl", "nhl", "running", "soccer", "tennis"]
    cities = ["atlanta", "austin", "baltimore", "birmingham", "boston", "buffalo", "charlotte", "chicago", "cincinnati", "cleveland", "columbus", "dallas", "denver", "detroit", "hartford", "houston", "indianapolis", "jacksonville", "kansascity", "lasvegas", "losangeles", "louisville", "memphis", "miami", "milwaukee", "minneapolis", "nashville", "neworleans", "nyc", "okc", "orlando", "philadelphia", "phoenix", "pittsburgh", "portland", "providence", "raleigh", "richmond", "rochester", "sacramento", "saltlakecity", "sanantonio", "sandiego", "sanfrancisco", "sanjose", "stlouis", "tampa", "virginiabeach", "washingtondc"]
    saved = {}
    with open(json_file) as data:
        line_num = 0
        for line in data:
            obj = json.loads(line)
            out = None
            subr = obj['subreddit'].lower()
            if subr in politics:
                out = "politics.csv"
            elif subr in sports:
                out = "sports.csv"
            elif subr in cities:
                out = "cities.csv"
            elif random.random() < prob:
                out = "noise.csv"
            if out != None and obj['author'] != "[deleted]":
                info = [
                    obj['id'],
                    subr,
                    obj['created_utc'],
                    obj['author'],
                    obj['link_id'],
                    obj['parent_id'],
                    obj['score'],
                    obj['body'].replace("\n", " ").replace("\r", "").replace("\t", " ")
                ]
                if out in saved:
                    saved[out].append(info)
                    if line_num % 10000 == 0:
                        print(out, len(saved[out]), line_num)
                else:
                    saved[out] = [info]
            line_num += 1
    for name in saved:
        header = not path.exists(name)
        with open(name, "a") as f:
            if header:
                f.write("id\tsubreddit\tcreated_utc\tauthor\tlink_id\tparent_id\tscore\tbody\n")
            for e in saved[name]:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7]))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_name = str(sys.argv[1])
        process(file_name)
