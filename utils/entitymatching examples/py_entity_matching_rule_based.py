import os
import py_entitymatching as em

# For further instructions on building rule-based matchers, please see the following link:
# http://anhaidgroup.github.io/py_entitymatching/v0.3.3/user_manual/matching.html#rule-based-matchers

# DBLP-Scholar rules: ['publisher_publisher_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5', 'title_title_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5', 'year_year_lev_sim(ltuple, rtuple) > 0.5', 'journal_journal_lev_sim(ltuple, rtuple) > 0.3']
# DBLP-ACM rules: ['title_title_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5', 'authors_authors_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5', 'year_year_lev_sim(ltuple, rtuple) > 0.5']
# Cameras/Shoes rules: ['title_title_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5']
# iTunes-Amazon rules: ['Song_Name_Song_Name_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5', 'Artist_Name_Artist_Name_cos_dlm_dc0_dlm_dc0 > 0.5', 'Album_Name_Album_Name_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5', 'Price_Price_cos_dlm_dc0_dlm_dc0 > 0.5']

datasets = ['Compas']
datasets_dir = 'data/'

for dataset in datasets:
    path_A = datasets_dir + os.sep + dataset + '/tableA.csv'
    path_B = datasets_dir + os.sep + dataset + '/tableB.csv'
    path_train = datasets_dir + os.sep + dataset + '/train.csv'
    path_test = datasets_dir + os.sep + dataset + '/test.csv'

    A = em.read_csv_metadata(path_A, key='id')
    B = em.read_csv_metadata(path_B, key='id')

    I = em.read_csv_metadata(path_train, key='id',
                             ltable=A, rtable=B,
                             fk_ltable='left_id', fk_rtable='right_id')

    J = em.read_csv_metadata(path_test, key='id',
                             ltable=A, rtable=B,
                             fk_ltable='left_id', fk_rtable='right_id')

    brm = em.BooleanRuleMatcher()

    atypes1 = em.get_attr_types(A)
    atypes2 = em.get_attr_types(B)
    block_c = em.get_attr_corres(A, B)
    tok = em.get_tokenizers_for_blocking()
    sim = em.get_sim_funs_for_blocking()

    F = em.get_features(A, B, atypes1, atypes2, block_c, tok, sim)

    print(F.feature_name.to_string())  # A set of similarity-based features are automatically created by the em

    # add rules here based on the set of generated features output in the last step
    brm.add_rule(['FullName_FullName_cos_dlm_dc0_dlm_dc0(ltuple, rtuple) > 0.5',
                  'Ethnic_Code_Text_Ethnic_Code_Text_exm(ltuple, rtuple) == 1'], F)

    predictions = brm.predict(J, target_attr='preds', append=True)
    predictions['preds'].to_csv(datasets_dir + os.sep + dataset + '/preds_brm.csv', index=False)
