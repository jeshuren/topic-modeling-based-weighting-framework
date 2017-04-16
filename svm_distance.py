#!/usr/bin/env python
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC


class SVMTrain:
    def __init__(self, options=None):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        if options is None:
            options = ''
        self._parse_options(options)

    def _set_to_default_values(self):
        self.kernel_type = 'rbf'
        self.degree = 3
        self.gamma = 'auto'
        self.C = 1.0
        self.cache_size = 8000
        self.weight = None
        self.probability = 0
        self.cross_validation = False
        self.nr_fold = 0
        self.output, self.ext = os.path.splitext(str(sys.argv[-1]))
        self.input = sys.argv[-1]

    @staticmethod
    def exit_with_help(argv):
        print("Usage: svm_train.py [options] dataset\n"
              "options:\n"
              "-t kernel_type : set type of kernel function (default 2)\n"
              "	0 -- linear: u'*v\n"
              "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
              "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
              "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
              "	4 -- precomputed kernel (kernel values in training_set_file)\n"
              "-d degree : set degree in kernel function (default 3)\n"
              "-g gamma : set gamma in kernel function (default 1/num_features)\n"
              "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
              "-m cachesize : set cache memory size in MB (default 3000)\n"
              "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, "
              "0 or 1 (default 0)\n"
              "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
              "-v n: n-fold cross validation mode\n"
              "-a auto tuning : set the parameter as true, when parameters select as automatically.\n"
              "-o output : set the name for model file".format(argv[0]))
        exit(1)

    def _parse_options(self, options):
        argc = len(options)
        if argc < 2:
            self.exit_with_help(options)

        if isinstance(options, list):
            argv = options
        elif isinstance(options, str):
            argv = options.split()
        else:
            raise TypeError("arg 1 should be a list or a str.")
        self._set_to_default_values()
        self._assign_weights(self.input)
        kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

        weight = {}
        i = 1
        while i < len(argv):
            if argv[i] == "-t":
                i += 1
                self.kernel_type = kernels[int(argv[i])]
            elif argv[i] == "-d":
                i += 1
                self.degree = int(argv[i])
            elif argv[i] == "-g":
                i += 1
                self.gamma = float(argv[i])
            elif argv[i] == "-m":
                i += 1
                self.cache_size = float(argv[i])
            elif argv[i] == "-c":
                i += 1
                self.C = float(argv[i])
            elif argv[i] == "-b":
                i += 1
                self.probability = int(argv[i])
            elif argv[i].startswith("-w"):
                weight_label = argv[i].split('-w')[-1]
                i += 1
                weight[int(weight_label)] = int(argv[i])
                self.weight.update(weight)
            elif argv[i] == "-v":
                i += 1
                self.cross_validation = 1
                self.nr_fold = int(argv[i])
                print ("Cross Validation...")
                if self.nr_fold < 2:
                    raise ValueError("n-fold cross validation: n must >= 2")
            elif argv[i] == "-o":
                i += 1
                self.output = str(argv[i])
            i += 1

    def _assign_weights(self, input_file):
        # assigning weights#
        data = np.genfromtxt(self.input,delimiter=',')
        self.X_train = data[:,0:data.shape[1]-2]
        self.y_train = data[:,data.shape[1]-1]
        positive = len(self.y_train[self.y_train == 1])
        negative = len(self.y_train[self.y_train == -1])
        if positive > negative:
            self.weight = {-1: abs(positive / negative) + 1, +1: abs(negative / positive) + 1}
        else:
            self.weight = {+1: abs(negative / positive) + 1, -1: abs(positive / negative) + 1}

    def _build_tuning_model(self):
        print ("building the model with auto tuning parameter...")
        data = np.genfromtxt(self.input,delimiter=',')
        self.X_train = data[:,0:data.shape[1]-2]
        self.y_train = data[:,data.shape[1]-1]
        # define the classifier#
        self.clf = SVC(C=self.C, kernel=self.kernel_type, degree=self.degree, gamma=self.gamma,
                       probability=self.probability, cache_size=self.cache_size, class_weight=self.weight)

        # tuning parameter ranges#
        c_param_range = [pow(2, -4), pow(2, -2), pow(2, 0), pow(2, 2), pow(2, 4)]
        g_param_range = [pow(2, -6), pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1)]
        if self.kernel_type in "linear":
            parameters = {'C': c_param_range}
        else:
            parameters = {'C': c_param_range, 'gamma': g_param_range}

        # tuning the parameter using grid search#
        self.clf = GridSearchCV(self.clf, parameters, n_jobs=4)
        # fit the model#
        self.clf.fit(self.X_train, self.y_train)
        self.prediction = self.clf.predict(self.X_train)
        self._f1_score()

    def _build_model(self):
        print ("building the model with default or manual parameters...")
        data = np.genfromtxt(self.input,delimiter=',')
        self.X_train = data[:,0:data.shape[1]-2]
        self.y_train = data[:,data.shape[1]-1]
        # define the classifier#
        self.clf = SVC(C=self.C, kernel=self.kernel_type, degree=self.degree, gamma=self.gamma,
                       probability=self.probability, cache_size=self.cache_size, class_weight=self.weight)
        # fit the model#
        self.clf.fit(self.X_train, self.y_train)
        self.prediction = self.clf.predict(self.X_train)
        self._f1_score()

    def _dump_distances(self):
        result = list(self.prediction)
        for i in range(len(result)):
            if list(self.y_train)[i] != self.prediction[i]:
                result[i] = "False"
            else:
                result[i] = "True"
        # Calculate the distances for every samples#
        self.distances = self.clf.decision_function(self.X_train)
        self.doc_id = [i + 1 for i in range(len(result))]
        distances = [abs(d) for d in self.distances]
        p_d = self._prob_distance(distances)
        # save the dataset sample distances#
        np.savetxt(self.output + "_report.txt",
                   np.c_[self.doc_id, self.y_train, self.prediction, self.distances, p_d],
                   fmt="%s", delimiter=',', comments='')
        # load the distances into pandas dataframe #
        df = pd.DataFrame(data=self.distances, index=self.doc_id, columns=["dist"])
        dist = df.sort_index(by=['dist'], ascending=[True])
        pos = dist[dist >= 0].dropna()
        neg = dist[dist < 0].dropna()
        if len(pos) > len(neg):
            self._process_min_maj(pos, neg)
        else:
            self._process_min_maj(neg, pos)

    def _process_min_maj(self, majority, minority):
        # calculate P(D) for majority samples#
        pd_maj = self._prob_distance(majority.values)
        self._write(zip(majority.index.values, pd_maj), "_maj")
        # calculate P(D) for minority samples#
        pd_min = self._prob_distance(minority.values)
        self._write(zip(minority.index.values, pd_min), "_min")

    def _write(self, index_distance, tag):
        # write an doc_id and P(D) values #
        i_d = sorted(index_distance)
        fname = self.output + "_P(D)" + tag + ".txt"
        out = open(fname, "w")
        out.write("doc_id" + "\t" + "P(D)" + tag + "\n")
        for i, d in i_d:
            out.write(str(i) + "\t" + str(d[0]) + "\n")
        out.close()

    @staticmethod
    def _prob_distance(distance):
        # Calculate P(D) #
        sum_dist = sum(distance)
        norm_dist = [float(i) / sum_dist for i in distance]
        return norm_dist

    def _cross_validation(self):
        scores = cross_val_score(self.clf, self.X_train, self.y_train, cv=self.nr_fold)
        print ("Cross-Validation accuracy:", scores)

    def _f1_score(self):
        fscore = f1_score(self.y_train, self.prediction, average='macro')
        print ("f1-score:", fscore)


if __name__ == "__main__":
    # create an object for SVMTrain#
    obj = SVMTrain(sys.argv)
    args = sys.argv
    if "-v" in args:
        if "-a" in args:
            obj._build_tuning_model()
            obj._cross_validation()
        else:
            obj._build_model()
            obj._cross_validation()
    elif "-a" in args:
        obj._build_tuning_model()
        obj._dump_distances()
    else:
        obj._build_model()
        obj._dump_distances()
