import itertools
import random
import string
import numpy as np
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


# to increase number rows increase 1000 or remove head(1000)
df = pd.read_csv("csranking.csv")
df = df.drop(df[df['scholarid'] == 'NOSCHOLARPAGE'].index)
counts=df.groupby('scholarid',as_index=False).size()
df = df.merge(counts, on="scholarid")
df=df.sort_values(by='size',ascending=False).head(1000)

col1 = ['left_' + col for col in df.columns]
col2 = ['right_' + col for col in df.columns]
col = col1 + col2
col.append('label')

li = []
for i in range(len(df.index)):
    for j in range(i, len(df.index)):
        item = [df.iloc[i].values.flatten().tolist(), df.iloc[j].values.flatten().tolist()]
        """
        Instructions for randomPerurbationNChar(item[m][n], k) function:
                        m=0->perturb left entities,
                        m=1->perturb right entities,
                        n=column to be perturbed,
                        k=number of chars to perturb
        """
        item[1][0] = randomPerurbationNChar(item[1][0], 2)
        item[1][1] = randomPerurbationNChar(item[1][1], 2)
        item[1][2] = randomPerurbationNChar(item[1][2], 10)

        # rows with equal scholarid are match
        if df.iloc[i]['scholarid'] == df.iloc[j]['scholarid']:
            item.append([1])
        else:
            item.append([0])
        li.append(list(itertools.chain(*item)))

df2 = pd.DataFrame(data=li, columns=col)
df2 = df2.drop(['left_scholarid', 'right_scholarid'], axis=1)

l_ids = list(range(0, len(df2)))
random.shuffle(l_ids)
r_ids = list(range(len(df2), len(df2) * 2))
random.shuffle(r_ids)
df2['left_id'] = l_ids
df2['right_id'] = r_ids

# Remove 90% of the non-matches to make the dataset more balanced
df2 = df2.drop(df2[df2['label'] == 0].sample(frac=.9).index)
print(df2['label'].value_counts())

df2[['left_id', 'left_name', 'left_institution', 'left_homepage', 'left_region', 'left_countryabbrv', 'left_Gender',
     'right_id', 'right_name', 'right_institution', 'right_homepage', 'right_region', 'right_countryabbrv',
     'right_Gender', 'label']].to_csv('faculty_match_for_em.csv', index=False)
