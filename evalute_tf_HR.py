import scipy.sparse as sp
import numpy as np
import math
import time
import re
from Recommendation import Recommendation
import os
import random


def Evaluate_HR(preditmatrix, rec, top_k):
    testmatrix = rec.testMatrix
    num_users, num_items = testmatrix.shape
    # 生成每个用户的推荐列表TOP-K { user_id:{item1,item2...},...}
    each_user_topK_item = dict()
    TOP_K = top_k  # 5，10

    for userid in range(0, num_users):
        user_u_vertor = list(preditmatrix[userid])
        if userid not in each_user_topK_item.keys():
            # each_user_topK_item[userid] = list(map(user_u_vertor.index, heapq.nlargest(TOP_K, user_u_vertor)))
            each_user_topK_item[userid] = np.argsort(user_u_vertor)[-TOP_K:]

    # 判断testmatrix中的元素是否在each_user_topK_item中出现
    num_testsample = sp.dok_matrix.count_nonzero(testmatrix)
    # print(preditmatrix)
    # print(each_user_topK_item)
    count = 0
    for (userid, itemid) in testmatrix.keys():
        if testmatrix[userid, itemid] >= rec.user_ave_rating_dict[userid]:
            if userid in each_user_topK_item.keys():
                if itemid in each_user_topK_item[userid]:
                    count += 1
        else:
            num_testsample -= 1
    HR = count / num_testsample
    return HR


def Generate_HR_resultfile(path, path_train, path_test, dataname):
    rec = Recommendation(path, path_train, path_test)
    top_k = 5
    result_file = os.getcwd() + '\\result\\' + dataname + '\\HR' + '_' + os.path.basename(
        path) + '.csv'
    with open(result_file, 'w') as result_f:
        result_f.write('Deep Matrix Factorization Models for Recommender Systems \n')
        filename = 'predictMatrix'
        result_f.write('num_user:%d\nnum_items:%d\nranting:%d\nSparsity level:%.3f\n' % (
            rec.num_users, rec.num_items, rec.num_rating, rec.num_rating / (rec.num_items * rec.num_users)))
        result_f.write("%9.9s\t%6.6s\n" % ('item_topk', 'HR'))
        while top_k <= 10:
            preditmatrix_bingxing = np.load(
                os.getcwd() + '\\out_file\\' + dataname + '\\' + filename + '_' + os.path.basename(
                    path_train) + '_DMF.npy')
            hr_result = Evaluate_HR(preditmatrix_bingxing, rec, top_k)
            line = "%9.9s\t%6.6s\n" % (top_k, str(hr_result))
            result_f.write(line)
            top_k += 5


if __name__ == '__main__':
    # test数据集合
    Hybird = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    Hybird_train = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_train.csv'
    Hybird_test = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_test.csv'

    # test_99_400
    test = os.getcwd() + '\\prepare_datasets\\test_99_400.base'
    test_train = os.getcwd() + '\\prepare_datasets\\test_99_400.base_train.csv'
    test_test = os.getcwd() + '\\prepare_datasets\\test_99_400.base_test.csv'

    # m1-100k
    ml_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv'
    ml_100k_train = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    ml_100k_test = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_test.csv'

    # m1-1m
    ml_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_train = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_test = os.getcwd() + '\\prepare_datasets\\ml-1m.test.rating'

    # HR
    Generate_HR_resultfile(test, test_train, test_test, 'DMF')