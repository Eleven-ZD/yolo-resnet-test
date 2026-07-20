from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    confusion_matrix

)



def evaluate_metric(
        labels,
        preds
):


    acc = accuracy_score(

        labels,

        preds

    )


    precision = precision_score(

        labels,

        preds

    )


    recall = recall_score(

        labels,

        preds

    )


    matrix = confusion_matrix(

        labels,

        preds

    )


    print(
        "Accuracy:",
        acc
    )


    print(
        "Precision:",
        precision
    )


    print(
        "Recall:",
        recall
    )


    print(
        "Confusion Matrix:"
    )


    print(matrix)