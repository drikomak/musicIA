import csv

def readCSV(filename:str) -> list[dict]:
    """Cette fonction ouvre un fichier CSV et le convertit en une liste de dictionnaires, qui ont pour clé les colonnes du CSV."""

    table = []
    file = open(filename+".csv", encoding="utf-8-sig", newline="\n")
    for ligne in csv.DictReader(file,delimiter=","):
        table.append(dict(ligne))
    file.close()

    return table

def toCSV(final,fileName):

    file = file=open(f"{fileName}.csv", "w", encoding="utf-8-sig")

    ajout_key=""
    for key in final[0].keys():
        ajout_key=f"{key};"

    file.write(ajout_key[:-1])

    for i in range(len(final)):
        ajout_value=""
        for value in final[i].values():
            ajout_value+=f"{value};"

        file.write("\n"+ajout_value[:-1])


table = readCSV("musiccaps-public-filter")
data = {}
somme = 0

for i in table:
    tags:str = i["aspect_list"]
    tags = tags.replace("[","")
    tags = tags.replace("]","")
    tags = tags.replace("'","")
    t = tags.split(", ")
    i["tags"] = t
    for j in t:
        if j not in data:
            data[j] = 0
        data[j] += 1
        somme +=1

print(dict(sorted(data.items(), key=lambda item: item[1], reverse=True)))
print(len(data), somme/len(data))
print(len(table))


p = 0
for i in table:
    for z in i["tags"]:
        if data[z] >= 20:
            p += 1
            break

print(p)

l = []
m = 0
d = []
for i in data:
    if data[i] >= 20:
        d.append(i)
        l.append({"id":m,"nom":i,"count":data[i]})
        m += 1
    

toCSV(l,"tags20")