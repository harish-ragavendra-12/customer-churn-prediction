from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model_training import train_models


def evaluate_models():

    (
        lr_model,
        rf_model,
        X_test,
        y_test
    ) = train_models()

    # Logistic Regression
    lr_predictions = lr_model.predict(X_test)

    print("\n")
    print("=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            lr_predictions
        )
    )

    print(
        classification_report(
            y_test,
            lr_predictions
        )
    )

    print(
        confusion_matrix(
            y_test,
            lr_predictions
        )
    )

    # Random Forest
    rf_predictions = rf_model.predict(X_test)

    print("\n")
    print("=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            rf_predictions
        )
    )

    print(
        classification_report(
            y_test,
            rf_predictions
        )
    )

    print(
        confusion_matrix(
            y_test,
            rf_predictions
        )
    )


if __name__ == "__main__":
    evaluate_models()