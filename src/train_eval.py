
"""LOSO cross-validation for Random Forest, SVM, Logistic Regression, and BiLSTM-Attention."""

import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout, Bidirectional, LSTM, Dense, Attention, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_evaluate_loso(data_dict, model_type='rf', window_size=20):
    """
    Leave-One-Subject-Out cross-validation.
    model_type: 'rf', 'svm', 'logreg', 'bilstm'
    Returns: (y_true_all, y_pred_all, per_subject_kappa)
    """
    all_y_true = []
    all_y_pred = []
    kappas = []
    subject_ids = sorted(set(k.split('_')[0] for k in data_dict.keys()))

    for test_subj in tqdm(subject_ids, desc=f"LOSO CV ({model_type})"):
        # Split data
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        for key, (feats, lbls) in data_dict.items():
            subj_id = key.split('_')[0]
            if subj_id == test_subj:
                X_test_list.append(feats)
                y_test_list.append(lbls)
            else:
                X_train_list.append(feats)
                y_train_list.append(lbls)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)

        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100,
                                           class_weight='balanced',
                                           random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_type == 'svm':
            model = SVC(kernel='rbf', C=10, gamma='scale',
                        class_weight='balanced', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_type == 'logreg':
            model = LogisticRegression(penalty='l2', C=1.0,
                                       class_weight='balanced',
                                       max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_type == 'bilstm':
            # Build sliding windows (20 epochs, label = central epoch)
            def create_sequences(data, labels, w=window_size):
                X_seq, y_seq = [], []
                for i in range(len(data) - w + 1):
                    X_seq.append(data[i:i+w])
                    y_seq.append(labels[i + w//2])
                return np.array(X_seq), np.array(y_seq)

            X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, window_size)
            X_te_seq, y_te_seq = create_sequences(X_test, y_test, window_size)

            # Model architecture (BiLSTM + Attention)
            inp = Input(shape=(window_size, 18))
            x = Conv1D(64, 3, padding='same', activation='relu')(inp)
            x = Dropout(0.25)(x)
            x = Conv1D(64, 3, padding='same', activation='relu')(x)
            x = Dropout(0.25)(x)
            x = Bidirectional(LSTM(64, return_sequences=True,
                                   kernel_regularizer='l2'))(x)
            x = Dropout(0.3)(x)
            attention = Attention()([x, x])
            x = Flatten()(attention)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            out = Dense(5, activation='softmax')(x)
            model = Model(inp, out)
            model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=15,
                                       restore_best_weights=True)
            model.fit(X_tr_seq, y_tr_seq,
                      validation_split=0.2,
                      epochs=100, batch_size=64,
                      callbacks=[early_stop], verbose=0)
            y_pred_seq = model.predict(X_te_seq, verbose=0).argmax(axis=1)

            # Align predictions with original test set (only central epochs)
            y_pred = np.full(len(y_test), -1, dtype=int)
            offset = window_size // 2
            y_pred[offset:offset + len(y_pred_seq)] = y_pred_seq
            valid_mask = y_pred != -1
            y_test = y_test[valid_mask]
            y_pred = y_pred[valid_mask]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        kappas.append(cohen_kappa_score(y_test, y_pred))

    return np.array(all_y_true), np.array(all_y_pred), np.array(kappas)