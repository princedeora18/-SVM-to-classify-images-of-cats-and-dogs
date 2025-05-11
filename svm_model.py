
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# train & evaluate SVM
def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel):
    print(f"Training SVM ({kernel})...")
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto'] if kernel == 'rbf' else ['auto']}
    clf = SVC(kernel=kernel, probability=True)
    grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
    print(f"Accuracy: {acc:.4f}\n")
    print(report)
    return model, acc, report
