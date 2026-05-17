class ConfusionMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.conf_mat = []
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self._build_confusion_matrix()
        self._compute_counts()

    def _build_confusion_matrix(self):
        for i in range(len(self.y_true)):
            if self.y_true[i] == 1 and self.y_pred[i] == 1:
                self.conf_mat.append("TP")
            elif self.y_true[i] == 0 and self.y_pred[i] == 0:
                self.conf_mat.append("TN")
            elif self.y_true[i] == 1 and self.y_pred[i] == 0:
                self.conf_mat.append("FN")
            else:
                self.conf_mat.append("FP")

    def _compute_counts(self):
        self.TP = self.conf_mat.count("TP")
        self.TN = self.conf_mat.count("TN")
        self.FP = self.conf_mat.count("FP")
        self.FN = self.conf_mat.count("FN")

    def accuracy(self):
        total = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / total if total else 0

    def recall(self):
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) else 0

    def precision(self):
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) else 0

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r) if (p + r) else 0


def run_test_case(name, y_true, y_pred):
    print(f"\n{name}")
    metrics = ConfusionMetrics(y_true, y_pred)

    print("TP:", metrics.TP, "FP:", metrics.FP,
          "FN:", metrics.FN, "TN:", metrics.TN)
    print("Confusion Metric:", metrics.conf_mat)
    print("Accuracy:", metrics.accuracy())
    print("Precision:", metrics.precision())
    print("Recall:", metrics.recall())
    print("F1 Score:", metrics.f1_score())


def main():
    # Test 0
    run_test_case(
        "Test 0",
        [1,0,0,1,0],
        [1,0,0,0,0]
    )
    # Test Case 1 — Perfect Classification
    run_test_case(
        "Test Case 1 — Perfect Classification",
        [1, 0, 1, 0],
        [1, 0, 1, 0]
    )

    # Test Case 2 — All Wrong
    run_test_case(
        "Test Case 2 — All Wrong",
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    )

    # Test Case 3 — No Predicted Positives
    run_test_case(
        "Test Case 3 — No Predicted Positives",
        [1, 0, 1, 0],
        [0, 0, 0, 0]
    )

    # Test Case 4 — No Actual Positives
    run_test_case(
        "Test Case 4 — No Actual Positives",
        [0, 0, 0, 0],
        [1, 0, 1, 0]
    )

    # Test Case - High Recall, Low Precison
    run_test_case(
        "Test Case  — High Recall, Low Precison",
        [1, 1, 0, 0, 0],
       [1, 1, 1, 1, 0]
    )

    # Test Case - High Precision, Low Recall
    run_test_case(
        "Test Case  — High Precision, Low Recall",
        [1, 1, 1, 0, 0],
       [1, 0, 0, 0, 0]
    )
    #Test -  data only contains very few instances of the thing you want to detect, and your classifier predicts negative all the time
    run_test_case(
        "Test Case  — Negative All Time",
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0]
    )

if __name__ == "__main__":
    main()