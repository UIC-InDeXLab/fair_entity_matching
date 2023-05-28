import os
import numpy as np
import py_entitymatching as em

# For further instructions on building non-neural matchers, please see the following link:
# http://anhaidgroup.github.io/py_entitymatching/v0.3.3/user_manual/matching.html#ml-matchers

datasets = ['Compas']
datasets_dir = 'data/'

for model in ['dt', 'rf', 'ln', 'lg', 'nb', 'svm']:
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

        # Create a set of ML-matchers
        if model == 'svm':
            model_ = em.SVMMatcher(name='SVM', random_state=0, probability=True)
        elif model == 'dt':
            model_ = em.DTMatcher(name='DecisionTree', random_state=0)
        elif model == 'rf':
            model_ = em.RFMatcher(name='RF', random_state=0)
        elif model == 'lg':
            model_ = em.LogRegMatcher(name='LogReg', random_state=0, max_iter=1000)
        elif model == 'ln':
            model_ = em.LinRegMatcher(name='LinReg')
        elif model == 'nb':
            model_ = em.NBMatcher(name='NB')

        atypes1 = em.get_attr_types(A)
        atypes2 = em.get_attr_types(B)

        block_c = em.get_attr_corres(A, B)
        tok = em.get_tokenizers_for_blocking()
        sim = em.get_sim_funs_for_blocking()
        F = em.get_features(A, B, atypes1, atypes2, block_c, tok, sim)

        print(F.feature_name)

        H = em.extract_feature_vecs(I,
                                    feature_table=F,
                                    attrs_after='label',
                                    show_progress=True)

        H = em.impute_table(H, missing_val=np.nan)

        model_.fit(table=H,
                   exclude_attrs=['id', 'left_id', 'right_id', 'label'],
                   target_attr='label')

        L = em.extract_feature_vecs(J, feature_table=F,
                                    attrs_after='label', show_progress=True)

        L = em.impute_table(L, missing_val=np.nan)

        predictions = model_.predict(table=L, exclude_attrs=['id', 'left_id', 'right_id', 'label'],
                                     append=True, target_attr='preds', inplace=False, return_probs=True,
                                     probs_attr='proba')

        print(predictions[['id', 'left_id', 'right_id', 'preds']])
        predictions['preds'].to_csv(datasets_dir + os.sep + dataset + '/preds_' + model + '.csv', index=False)
        eval_result = em.eval_matches(predictions, 'label', 'preds')
        em.print_eval_summary(eval_result)
