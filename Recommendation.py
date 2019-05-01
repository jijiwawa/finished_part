# -*- coding: utf-8 -*-
import scipy.sparse as sp
import numpy as np
import math
import time
import re
import os
import random


class Recommendation():

    # 每个file 均为路径
    def __init__(self, file,train_file,test_file):
        # 获取用户数和物品数量
        self.num_users, self.num_items ,self.num_rating= self.Count_Num_Of_UserAndItem(file)
        print('用户数：',self.num_users)
        print('物品数：',self.num_items)
        print('评分数：',self.num_rating)
        # 构建训练评分矩阵
        self.trainMatrix = self.Transform_csv_To_RatingMatrix(train_file)
        # 构建测试评分矩阵用于后期评估运算
        self.testMatrix = self.Transform_csv_To_RatingMatrix(test_file)
        # 评分矩阵参数
        self.ratingMax, self.ratingMim, self.num_scale = 5, 1, 5
        # 生成每个用户评分了的物品集合 {user:[item1,item2]}
        self.coItemDict = self.Generate_ratingItemDict_ForEachUser(self.trainMatrix)
        # 每个用户评分均值
        print('用户评分了的物品集合：')
        print(self.coItemDict)

    # 统计总样本中用户数和物品数
    def Count_Num_Of_UserAndItem(self, ratedfile):
        num_users, num_items = 0, 0
        count = 0
        with open(ratedfile, "r") as f:
            line = f.readline()
            while line != None and line != "":
                # [\s,:]
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId = int(float(arr[0])), int(float(arr[1]))
                num_users = max(num_users, userId)
                num_items = max(num_items, itemId)
                count += 1
                line = f.readline()
        return num_users + 1, num_items + 1,count

    def Transform_list_To_RatingMatrix(self, train_list):
        Matrix = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        for line in train_list:
            userId, itemId, rating = int(float(line[0])), int(float(line[1])), int(float(line[2]))
            Matrix[userId, itemId] = rating
        return Matrix

    # 将训练样本转换成评分矩阵
    def Transform_csv_To_RatingMatrix(self, file):
        ratingMatrix = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                ratingMatrix[userId, itemId] = rating
                line = f.readline()
        return ratingMatrix

    # 分割数据集为测试集和训练集并写入train 和 test文件
    def SplitData(self, file, M, k, seed):
        train, test = [], []
        random.seed(seed)
        # 以列表的形式存
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                if random.randint(0, M) == k:
                    test.append([arr[0], arr[1], arr[2]])
                else:
                    train.append([arr[0], arr[1], arr[2]])
                line = f.readline()
        return train, test

    # preditmatrix 是一个array[][]
    # testmatrix 是一个dokmatirx
    # 从评分矩阵中生成用户的评分了的物品列表 用户/物品
    def Generate_ratingItemDict_ForEachUser(self, trainMatrix):
        ratingItemList = dict()
        for (userid, itemid) in trainMatrix.keys():
            if userid not in ratingItemList.keys():
                ratingItemList[userid] = set()
            ratingItemList[userid].add(itemid)
        return ratingItemList

    # 从csv中生成物品的评分字典  {物品id:{评分}}
    def Generate_ratingDict_ForEachItem(self, file):
        itemRatingDict = dict()
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                if itemId not in itemRatingDict.keys():
                    itemRatingDict[itemId] = []
                itemRatingDict[itemId].append(rating)
                line = f.readline()
        return itemRatingDict

    # 从评分矩阵中生成用户的评分了的物品列表 用户/物品
    def Generate_ratingItemDict_ForEachUser(self, trainMatrix):
        ratingItemList = dict()
        for (userid, itemid) in trainMatrix.keys():
            if userid not in ratingItemList.keys():
                ratingItemList[userid] = set()
            ratingItemList[userid].add(itemid)
        return ratingItemList


def SplitData_To_TrainandTest(datafile, M, k, seed):
    random.seed(seed)
    # 以文件的形式存储
    trainfile_path = os.getcwd() + '\\prepare_datasets\\' + os.path.basename(datafile) + '_train.csv'
    if os.path.exists(trainfile_path):
        os.remove(trainfile_path)
    f_train = open(trainfile_path, 'a')
    testfile_path = os.getcwd() + '\\prepare_datasets\\' + os.path.basename(datafile) + '_test.csv'
    if os.path.exists(testfile_path):
        os.remove(testfile_path)
    f_test = open(testfile_path, 'a')
    with open(datafile, "r") as f:
        line = f.readline()
        while line != None and line != "":
            if random.randint(0, M) == k:
                f_test.write(line)
            else:
                f_train.write(line)
            line = f.readline()
    f_train.close()
    f_test.close()



if __name__ == '__main__':
    s=1
