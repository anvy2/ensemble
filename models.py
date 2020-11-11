from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from mlens.ensemble import SuperLearner, SequentialEnsemble
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import confusion_matrix, classification_report


class Models:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def SVM_Classifier(self):
        svm = SVC(gamma='scale', probability=True)
        svm.fit(self.X_train, self.y_train)
        svm_pred = svm.predict(self.X_test)
        score = accuracy_score(self.y_test, svm_pred)
        print("SVM accuracy is :{}".format(score))

    def RF_Classifier(self, estimators):
        rf = RandomForestClassifier(n_estimators=estimators)
        rf.fit(self.X_train, self.y_train)
        rf_pred = rf.predict(self.X_test)
        score = accuracy_score(self.y_test, rf_pred)
        print("random forest accuracy is :{}".format(score))

    def LR_Classifier(self):
        log = LogisticRegression(solver='liblinear')
        log.fit(self.X_train, self.y_train)
        pred = log.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("LogisticRegression accuracy is :{}".format(score))

    def DT_Classifier(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.y_train)
        pred = dt.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("DecisionTreeClassifier accuracy is :{}".format(score))

    def NB_Classifier(self):
        guassian = GaussianNB()
        guassian.fit(self.X_train, self.y_train)
        pred = guassian.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("GaussianNB accuracy is :{}".format(score))

    def KNN_Classifier(self):
        KNN = KNeighborsClassifier()
        KNN.fit(self.X_train, self.y_train)
        pred = KNN.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("KNeighborsClassifier accuracy is :{}".format(score))

    def AdaBoost_Classifier(self):
        Ada = AdaBoostClassifier()
        Ada.fit(self.X_train, self.y_train)
        pred = Ada.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("AdaBoostClassifier accuracy is :{}".format(score))

    def Bagging_Classifier(self, estimators):
        Bagging = BaggingClassifier(n_estimators=estimators)
        Bagging.fit(self.X_train, self.y_train)
        pred = Bagging.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("BaggingClassifier accuracy is :{}".format(score))

    def ExtraTress_Classifier(self, estimators):
        Ex_Tree = ExtraTreesClassifier(n_estimators=estimators)
        Ex_Tree.fit(self.X_train, self.y_train)
        pred = Ex_Tree.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("ExtraTreesClassifier accuracy is :{}".format(score))

    def XtremeGB_Classifier(self):
        XGB = XGBClassifier()
        XGB.fit(self.X_train, self.y_train)
        pred = XGB.predict(self.X_test)
        score = accuracy_score(self.y_test, pred)
        print("XGBClassifier accuracy is :{}".format(score))

    def get_super_learner_models(self):
        models = list()
        models.append(LogisticRegression(solver='liblinear'))
        models.append(DecisionTreeClassifier())
        models.append(SVC(gamma='scale', probability=True))
        models.append(GaussianNB())
        models.append(KNeighborsClassifier())
        models.append(AdaBoostClassifier())
        models.append(BaggingClassifier(n_estimators=300))
        models.append(RandomForestClassifier(n_estimators=300))
        models.append(ExtraTreesClassifier(n_estimators=300))
        models.append(XGBClassifier())
        return models

    def get_super_learner(self, meta):
        ensemble = SequentialEnsemble(scorer=accuracy_score, shuffle=True,
                                      model_selection=True, backend='threading', sample_size=len(self.X_train))
        # ensemble = SuperLearner(scorer=accuracy_score,
        #                         folds=10, shuffle=True, model_selection=True, sample_size=len(self.X_train))
        models = self.get_super_learner_models()
        ensemble.add('blend', models)
        ensemble.add('stack', models)
        ensemble.add('subsemble', models)
        ensemble.add_meta(meta)
        return ensemble

    def mlen_combined_model(self, meta):
        ensemble = self.get_super_learner(meta)
        ensemble.fit(self.X_train.values, self.y_train.values)
        print(ensemble.data)

        pred = ensemble.predict(self.X_test.values)
        score = accuracy_score(self.y_test, pred)
        print("Super Learner accuracy is :{}".format(score))
        return pred

    def final_report(self, pred):
        print(classification_report(self.y_test, pred))
