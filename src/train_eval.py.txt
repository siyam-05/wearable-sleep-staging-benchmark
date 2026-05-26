import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from scipy.stats import ttest_rel
import pandas as pd

def train_eval_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'confusion': confusion_matrix(y_test, y_pred),
        'model': rf,
        'y_pred': y_pred
    }

def train_eval_svm(X_train, y_train, X_test, y_test):
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'model': svm,
        'y_pred': y_pred
    }

def train_eval_lr(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'model': lr,
        'y_pred': y_pred
    }

def loso_evaluation(DATA, model_func, model_name):
    subjects = list(DATA.keys())
    accs, kappas = [], []
    for test_subj in subjects:
        X_train = np.vstack([DATA[s][0] for s in subjects if s != test_subj])
        y_train = np.concatenate([DATA[s][1] for s in subjects if s != test_subj])
        X_test = DATA[test_subj][0]
        y_test = DATA[test_subj][1]
        res = model_func(X_train, y_train, X_test, y_test)
        accs.append(res['accuracy'])
        kappas.append(res['kappa'])
    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'kappa_mean': np.mean(kappas),
        'kappa_std': np.std(kappas),
        'accs': accs,
        'kappas': kappas
    }

def statistical_comparison(rf_kappas, svm_kappas, lr_kappas):
    t_rf_svm, p_rf_svm = ttest_rel(rf_kappas, svm_kappas)
    t_rf_lr, p_rf_lr = ttest_rel(rf_kappas, lr_kappas)
    return {
        'rf_vs_svm': {'t': t_rf_svm, 'p': p_rf_svm},
        'rf_vs_lr': {'t': t_rf_lr, 'p': p_rf_lr}
    }