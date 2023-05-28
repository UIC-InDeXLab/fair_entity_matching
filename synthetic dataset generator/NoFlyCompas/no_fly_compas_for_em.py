import itertools
import random
import string

import pandas as pd


def randomlyChangeNChar(word, value):
    length = len(word)
    word = list(word)
    k = random.sample(range(0, length), value)
    for index in k:
        word[index] = random.choice(string.ascii_lowercase)
    return "".join(word)


def randomlyAddNChar(word, value):
    length = len(word)
    word = list(word)
    k = random.sample(range(0, length), value)
    for index in k:
        word.insert(index, random.choice(string.ascii_lowercase))
    return "".join(word)


def randomlyRemoveNChar(word, value):
    length = len(word)
    word = list(word)
    k = random.sample(range(0, length), value)
    k.sort(reverse=True)
    for index in k:
        word.pop(index)
    return "".join(word)


def randomPerurbationNChar(word, value):
    list1 = [1, 2, 3]
    index = random.choice(list1)
    if index == 1:
        return randomlyChangeNChar(word, value)
    if index == 2:
        return randomlyAddNChar(word, value)
    if index == 3:
        return randomlyRemoveNChar(word, value)


df = pd.read_csv("compas-scores-raw.csv")
df['FullName']= df[['FirstName', 'LastName']].agg(' '.join, axis=1)
df = df[['Person_ID', 'FullName', 'Ethnic_Code_Text']]
df = df[(df['Ethnic_Code_Text'] == "Caucasian") | (df['Ethnic_Code_Text'] == "African-American")].sample(n=4000)
df_criminal = df.sample(n=2000)
df_population_white = df[df['Ethnic_Code_Text'] == "Caucasian"].sample(n=1600)
df_population_black = df[df['Ethnic_Code_Text'] == "African-American"].sample(n=400)
population = [df_population_white, df_population_black]
df_population = pd.concat(population)
print(df_population.value_counts('Ethnic_Code_Text'))

col1 = ['left_' + col for col in df.columns]
col2 = ['right_' + col for col in df.columns]
col = col1 + col2
col.append('label')

li = []
for i in range(len(df_criminal.index)):
    for j in range(i, len(df_population.index)):
        item = [df_criminal.iloc[i].values.flatten().tolist(), df_population.iloc[j].values.flatten().tolist()]
        item[1][1] = randomPerurbationNChar(item[1][1], 1)

        if df_criminal.iloc[i]['Person_ID'] == df_population.iloc[j]['Person_ID']:
            item.append([1])
        else:
            item.append([0])
        li.append(list(itertools.chain(*item)))

df2 = pd.DataFrame(data=li, columns=col)
df2 = df2.drop(['left_Person_ID', 'right_Person_ID'], axis=1)

l_ids = list(range(0, len(df2)))
random.shuffle(l_ids)
r_ids = list(range(len(df2), len(df2) * 2))
random.shuffle(r_ids)
df2['left_id'] = l_ids
df2['right_id'] = r_ids

# Remove 95% of the non-matches to make the dataset more balanced
df2 = df2.drop(df2[df2['label'] == 0].sample(frac=0.95).index)
print(df2['label'].value_counts())

df2 = df2[['left_id', 'left_FullName', 'left_Ethnic_Code_Text', 'right_id', 'right_FullName', 'right_Ethnic_Code_Text', 'label']]
df2.to_csv('compas_for_em.csv', index=False)
