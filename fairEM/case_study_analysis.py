import pandas as pd

# This code can be used to compare the left and right entities for different cases such as TP, FP, etc.


dir = 'data/DeepMatcher/iTunes-Amazon/test.csv'
preds = 'data/Ditto/iTunes-Amazon/preds.csv'
df = pd.read_csv(dir)
preds = pd.read_csv(preds).values.tolist()

for index, row in df.iterrows():
    if preds[index][0] == 1 and row['label'] == 1:  # Adjust the if condition accordingly for the case interested, e.g: preds==1 AND label==1 means TP.
        print('song:', row['left_Song_Name'], '|', row['right_Song_Name'])
        print('artist:', row['left_Artist_Name'], '|', row['right_Artist_Name'])
        print('album:', row['left_Album_Name'], '|', row['right_Album_Name'])
        print('genre:', row['left_Genre'], '|', row['right_Genre'])
        print('price:', row['left_Price'], '|', row['right_Price'])
        print('copyright:', row['left_CopyRight'], '|', row['right_CopyRight'])
        print('duration:', row['left_Time'], '|', row['right_Time'])
        print('released:', row['left_Released'], '|', row['right_Released'])
        print("----------------------------------------------------------")

# dir = 'data/DeepMatcher/DBLP-ACM/test.csv'
# preds = 'data/HierMatcher/DBLP-ACM/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if preds[index][0] ==1 and row['label']==0:
#         print('title:', row['left_title'], '|', row['right_title'])
#         print('author:', row['left_authors'], '|', row['right_authors'])
#         print('venue:', row['left_venue'], '|', row['right_venue'])
#         print('year:', row['left_year'], '|', row['right_year'])
#         print("----------------------------------------------------------")


# dir = 'data/DeepMatcher/DBLP-Scholar/test.csv'
# preds = 'data/GNEM/DBLP-Scholar/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if  preds[index][0] == row['label']:
#         print('publisher:', row['left_publisher'], '|', row['right_publisher'])
#         print('title:', row['left_title'], '|', row['right_title'])
#         print('author:', row['left_author'], '|', row['right_author'])
#         print('year:', row['left_year'], '|', row['right_year'])
#         print('entry_type:', row['left_ENTRYTYPE'], '|', row['right_ENTRYTYPE'])
#         print('journal:', row['left_journal'], '|', row['right_journal'])
#         print('number:', row['left_number'], '|', row['right_number'])
#         print('volume:', row['left_volume'], '|', row['right_volume'])
#         print('pages:', row['left_pages'], '|', row['right_pages'])
#         print('booktitle:', row['left_booktitle'], '|', row['right_booktitle'])
#         print("----------------------------------------------------------")

#
# dir = 'data/DeepMatcher/Cameras/test.csv'
# preds = 'data/DecisionTree/Cameras/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if preds[index][0] == 0 and row['label'] == 1:
#         print('title:', row['left_title'], '||', row['right_title'])

# dir = 'data/DeepMatcher/Shoes/test_others.csv'
# preds = 'data/GNEM/Shoes/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if row['left_locale'] != row['right_locale'] and preds[index][0] == row['label']:
#         print('title:', row['left_title'], '|', row['right_title'])


#
# dir = 'data/DeepMatcher/CSRankings/test.csv'
# preds = 'data/Ditto/CSRankings/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if preds[index][0] ==0 and row['label']==1:
#         print('name:', row['left_name'], '|', row['right_name'])
#         print('countryabbrv:', row['left_countryabbrv'], '|', row['right_countryabbrv'])
#         print("----------------------------------------------------------")


# dir = 'data/DeepMatcher/Compas/test.csv'
# preds = 'data/DeepMatcher/Compas/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if preds[index][0] == 1 and row['label'] == 0 and row['left_Ethnic_Code_Text'] == row['right_Ethnic_Code_Text']:
#         print('fullName:', row['left_FullName'], '|', row['right_FullName'])
#         print('Ethnic_Code:', row['left_Ethnic_Code_Text'], '|', row['right_Ethnic_Code_Text'])
#         print("----------------------------------------------------------")
