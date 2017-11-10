require(["nbextensions/snippets_menu/main"], function (snippets_menu) {
    console.log('Loading `snippets_menu` customizations from `custom.js`');
    var my_favorites = {
            {
                'name': 'PrepareData',
                'sub-menu': [
                    {
                        'name' : 'load_iris',
                        'snippet' : [
                            "from sklearn.datasets import load_iris",
                            "from sklearn.model_selection import train_test_split",
                            "iris = load_iris()",
                            "X = iris.data",
                            "y = iris.target",
                            "X_train, X_test, y_train, y_test = train_test_split(",
                            "    X, y, test_size=0.3, random_state=12, stratify=y)",
                        ],
                    },
                    {
                        'name' : 'load_digit',
                        'snippet' : [
                            "from sklearn.datasets import load_digits",
                            "from sklearn.model_selection import train_test_split",
                            "digits = load_digits()",
                            "X = digits.data",
                            "y = digits.target",
                            "X_train, X_test, y_train, y_test = train_test_split(",
                            "    X, y, test_size=0.3, random_state=12, stratify=y)",
                        ],
                    },
                ],
            },
            {
                'name': 'pandas',
                'sub-menu': [
                    {
                        'name' : 'DataFrame Check',
                        'snippet' : [
                            "print(df.head())",
                            "print(\"==========\")",
                            "print(df.describe())",
                            "print(\"==========\")",
                            "print(df.info())",
                        ],
                    },
                ],
            },
            {
                'name': 'sklearn',
                'sub-menu': [
                    {
                        'name' : 'LogisticRegression',
                        'snippet' : [
                            "from sklearn.linear_model import LogisticRegression",
                            "lr = LogisticRegression()",
                            "lr.fit(X_train, y_train)",
                            "print(lr.score(X_test, y_test))",
                            "y_predict = lr.predict(X_test)",
                        ],
                    },
                    {
                        'name' : 'SGDClassifier',
                        'snippet' : [
                            "from sklearn.linear_model import SGDClassifier",
                            "sgd = SGDClassifier(loss=\"hinge\", penalty=\"l2\")",
                            "sgd.fit(X_train, y_train)",
                            "print(sgd.score(X_test, y_test))",
                            "y_predict = sgd.predict(X_test)"
                        ],
                    },
                    {
                        'name' : 'DecisionTreeClassifier',
                        'snippet' : [
                            "from sklearn.tree import DecisionTreeClassifier",
                            "from sklearn.tree import export_graphviz",
                            "import graphviz",
                            "tree = DecisionTreeClassifier(random_state=12, max_depth=5)",
                            "tree.fit(X_train, y_train)",
                            "print(tree.score(X_test, y_test))",
                            "y_predict = tree.predict(X_test)",
                            "export_graphviz(tree, out_file='graph/tree.dot', class_names=['Negative', 'Positive']",
                            "                , feature_names=X_train.columns, impurity=False, filled=True)",
                            "with open('graph/tree.dot') as f:",
                            "    dot_graph = f.read()",
                            "graphviz.Source(dot_graph)"
                        ],
                    },
                    {
                        'name' : 'MLPClassifier',
                        'snippet' : [
                            "from sklearn.neural_network import MLPClassifier",
                            "",
                            "tuned_parameters = {",
                            "    'hidden_layer_sizes': (100,),",
                            "    'activation': 'relu',",
                            "    'solver': 'sgd',",
                            "    'alpha': 0.0001,",
                            "    'batch_size': 'auto',",
                            "    'learning_rate': 'constant',",
                            "    'learning_rate_init': 0.001,",
                            "    'power_t': 0.5,",
                            "    'max_iter': 10000,",
                            "    'shuffle': True,",
                            "    'random_state': 0,",
                            "    'tol': 1e-4,",
                            "    'verbose': True,",
                            "    'warm_start': False,",
                            "    'momentum': 0.9,",
                            "    'nesterovs_momentum': True,",
                            "    'early_stopping': False,",
                            "    'validation_fraction': 0.1,",
                            "    'beta_1': 0.9,",
                            "    'beta_2': 0.99,",
                            "    'epsilon': 1e-8,",
                            "}",
                            "",
                            "clf = MLPClassifier(**tuned_parameters)",
                            "clf.fit(X_train, y_train)",
                            "y_predict = clf.predict(X_test)",
                            "print (clf.score(X_test, y_test))"
                        ],
                    },
                    {
                        'name' : 'SVC',
                        'snippet' : [
                            "from sklearn.svm import SVC",
                            "",
                            "tuned_parameters = {",
                            "    'kernel': 'rbf',",
                            "    'gamma': 1e-3, ",
                            "    'C': 10,",
                            "}",
                            "",
                            "svc = SVC(**tuned_parameters)",
                            "svc.fit(X_train, y_train)",
                            "y_predict = svc.predict(X_test)",
                            "print(svc.score(X_test, y_test))"
                        ],
                    },
                    {
                        'name' : 'StandardScaler',
                        'snippet' : [
                            "from sklearn.preprocessing import StandardScaler",
                            "sc = StandardScaler()",
                            "sc.fit(X_train)",
                            "X_train_std = sc.transform(X_train)",
                            "sc.fit(X_test)",
                            "X_test_std = sc.transform(X_test)",
                        ],
                    },
                    {
                        'name' : 'train_test_split',
                        'snippet' : [
                            "from sklearn.model_selection import train_test_split",
                            "X_split, X_val, y_split, y_val = train_test_split(",
                            "    X_train, y_train, test_size=0.3, random_state=12, stratify=y_train)"
                        ],
                    },
                    {
                        'name' : 'GridSearch',
                        'snippet' : [
                            "from sklearn.model_selection import GridSearchCV",
                            "from sklearn.svm import SVC",
                            "from pandas import DataFrame",
                            "from sklearn.metrics import classification_report",
                            "",
                            "tuned_parameters = [",
                            "    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C':[1, 10, 100, 1000]}",
                            "    , {'kernel': ['linear'], 'C':[1, 10, 100, 1000]}",
                            "]",
                            "scores = ['accuracy'] ",
                            "",
                            "for score in scores:",
                            "    print('\\n' + '='*50)",
                            "    print(score)",
                            "    print('='*50)",
                            "    ",
                            "    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring=score, n_jobs=-1)",
                            "    clf.fit(X_train, y_train)",
                            "    ",
                            "    print(\"\\n+ Best Parameter:\\n\")",
                            "    print(clf.best_estimator_)",
                            "    ",
                            "    print(\"\\n+ トレーニングデータでcvした時の平均スコア:\\n\")",
                            "    df = DataFrame(clf.cv_results_)",
                            "    print(df[['rank_test_score', 'params', 'mean_test_score', 'std_test_score' ]].to_csv(sep='\\t'))",
                            "    ",
                            "    print(\"\\n+ テストデータでの識別結果:\\n\")",
                            "    y_true, y_pred = y_test, clf.predict(X_test)",
                            "    print(classification_report(y_true, y_pred))",
                        ],
                    },
                ],
            },
            {
                'name': 'matplotlib',
                'sub-menu': [
                    {
                        'name' : 'Setup for notebook',
                        'snippet' : [
                            "from __future__ import print_function, division",
                            "import numpy as np",
                            "import pandas as pd",
                            "import matplotlib as mpl",
                            "from matplotlib import pyplot as plt, rcParams",
                            "%matplotlib inline",
                            "plt.style.use('ggplot')",
                            "# 黒背景だと見にくいのでstyleを指定",
                            "# mpl.style.use('seaborn-white')",
                            "# 日本語フォントを指定する",
                            "rcParams['font.family']='IPAGothic'",
                        ],
                    },
                ],
            },
            {
                'name': 'MagicCommand',
                'sub-menu': [
                    {
                        'name' : 'timeit on a cell',
                        'snippet' : [
                            "%%timeit -n 1000 -r 3",
                            "",
                            "for i in range(1000):",
                            "    i*2",
                        ],
                    },
                ],
            },
        ],
    };
    snippets_menu.options['menus'].push(snippets_menu.default_menus[0]);
    snippets_menu.options['menus'].push(my_favorites);
    console.log('Loaded `snippets_menu` customizations from `custom.js`');
});

