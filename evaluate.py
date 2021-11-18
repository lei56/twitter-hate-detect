from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate(y_test, y_pred):
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy score:")
    print(accuracy_score(y_test, y_pred))