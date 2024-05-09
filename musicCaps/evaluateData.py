import csv

def readCSV(filename:str) -> list[dict]:
    """Cette fonction ouvre un fichier CSV et le convertit en une liste de dictionnaires, qui ont pour clé les colonnes du CSV."""

    table = []
    file = open(filename+".csv", encoding="utf-8-sig", newline="\n")
    for ligne in csv.DictReader(file,delimiter=";"):
        table.append(dict(ligne))
    file.close()

    return table


table = readCSV("musiccaps-public")
data = {}
somme = 0

for i in table:
    tags:str = i["aspect_list"]
    tags = tags.replace("[","")
    tags = tags.replace("]","")
    tags = tags.replace("'","")
    t = tags.split("; ")
    for j in t:
        if j not in data:
            data[j] = 0
        data[j] += 1
        somme +=1

print(dict(sorted(data.items(), key=lambda item: item[1], reverse=True)))
print(len(data), somme/len(data))