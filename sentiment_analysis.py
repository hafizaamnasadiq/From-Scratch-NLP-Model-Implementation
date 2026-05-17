import csv
import html
import math
import re
import sys
import time
from collections import Counter


class EvaluationMetrics:
    # This class calculates the evaluation scores for our model.
    # It uses 1 for positive reviews and 0 for negative reviews.

    # This function counts TP, FP, FN, and TN.
    def confusion_matrix(self, y_true, y_pred):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                fn += 1
            else:
                tn += 1

        return tp, fp, fn, tn

    # This function calculates how many predictions were correct.
    def accuracy(self, y_true, y_pred):
        correct = 0

        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1

        return correct / len(y_true)

    # This function calculates precision.
    # Precision answers: when the model says positive, how often is it correct?
    def precision(self, y_true, y_pred):
        tp, fp, fn, tn = self.confusion_matrix(y_true, y_pred)

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    # This function calculates recall.
    # Recall answers: out of all real positive reviews, how many did the model find?
    def recall(self, y_true, y_pred):
        tp, fp, fn, tn = self.confusion_matrix(y_true, y_pred)

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    # This function calculates F1 score.
    # F1 score combines precision and recall.
    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)


class SentimentAnalysis:

    # It loads data, cleans text, trains Logistic Regression, and evaluates results.
    # This function sets the starting values for the project.
    def __init__(self):
        self.max_words = 5000
        self.learning_rate = 0.005
        self.epochs = 5

        # We use the first 1000 reviews from the CSV file in order.
        # We do not shuffle the dataset.
        self.data_limit = 1000

        # Error analysis will show the first 40 reviews from the test set.
        # Since the test set starts at CSV row 801, this checks rows 801 to 840.
        self.error_analysis_limit = 40

        self.vocabulary = {}
        self.weights = []
        self.bias = 0.0

        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "while",
            "of", "to", "in", "on", "for", "with", "as", "at",
            "by", "from", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being",
            "it", "its", "i", "you", "he", "she", "we", "they",
            "them", "his", "her", "their", "our", "my", "your",
            "me", "him", "so", "very", "too", "just", "br"
        }

    # This helper function calculates how long a part of the program took.
    def get_time_taken(self, start_time):
        end_time = time.perf_counter()
        return end_time - start_time

    # This function loads the IMDb CSV file.
    # It loads the first 1000 reviews in the original CSV order.
    # It does not shuffle the data.Data is in order.
    def load_data(self, filename, limit=1000):
        csv.field_size_limit(sys.maxsize)

        data = []

        with open(filename, "r", encoding="utf-8", errors="replace") as file:
            reader = csv.DictReader(file)

            for row in reader:
                review = row["review"]
                sentiment = row["sentiment"].strip().lower()

                if sentiment == "positive":
                    label = 1
                else:
                    label = 0

                data.append((review, label))

                if len(data) == limit:
                    break

        return data

    # This function splits the data into training data and test data.
    def split_data(self, data):
        split_point = int(len(data) * 0.8)

        train_data = data[:split_point]
        test_data = data[split_point:]

        return train_data, test_data

    # This function cleans one review.
    # It removes HTML symbols, HTML tags, capital letters, punctuation, and extra spaces.
    # I have re-used preprocessing function from the previous Naive Bayes assignment.
    def clean_text(self, text):
        text = html.unescape(text)
        text = re.sub(r"<.*?>", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # This function turns cleaned review text into useful words.
    # It removes very short words and stop words.
    # Example: "This movie was amazing!" becomes ("movie", "amazing").
    def tokenize(self, text):
        text = self.clean_text(text)
        words = text.split()

        tokens = []

        for word in words:
            if len(word) < 2:
                continue

            if word in self.stop_words:
                continue

            tokens.append(word)

        return tokens

    # This function builds the vocabulary from training reviews only.
    def build_vocabulary(self, train_reviews):
        word_counts = Counter()

        for review in train_reviews:
            tokens = self.tokenize(review)
            word_counts.update(tokens)

        most_common_words = word_counts.most_common(self.max_words)

        for index, word_and_count in enumerate(most_common_words):
            word = word_and_count[0]
            self.vocabulary[word] = index

        self.weights = [0.0] * len(self.vocabulary)
        self.bias = 0.0

    # This function changes one review into numbers using Bag-of-Words.
    def review_to_vector(self, review):
        tokens = self.tokenize(review)
        vector = {}

        for word in tokens:
            if word in self.vocabulary:
                index = self.vocabulary[word]
                vector[index] = vector.get(index, 0) + 1

        return vector

    # This function changes a score into a probability between 0 and 1.
    def sigmoid(self, score):
        if score > 500:
            score = 500
        elif score < -500:
            score = -500

        return 1 / (1 + math.exp(-score))

    # This function predicts the probability that one review is positive.
    def predict_probability(self, vector):
        score = self.bias

        for index, count in vector.items():
            score += self.weights[index] * count

        return self.sigmoid(score)

    # This function trains the Logistic Regression model.
    def train_model(self, train_reviews, train_labels):
        train_vectors = []

        for review in train_reviews:
            train_vectors.append(self.review_to_vector(review))

        for epoch in range(self.epochs):
            epoch_start_time = time.perf_counter()
            total_loss = 0.0

            for i in range(len(train_vectors)):
                vector = train_vectors[i]
                true_label = train_labels[i]

                probability = self.predict_probability(vector)
                error = probability - true_label

                for index, count in vector.items():
                    self.weights[index] -= self.learning_rate * error * count

                self.bias -= self.learning_rate * error

                safe_probability = max(min(probability, 0.999999), 0.000001)

                loss = -(
                    true_label * math.log(safe_probability)
                    + (1 - true_label) * math.log(1 - safe_probability)
                )

                total_loss += loss

            average_loss = total_loss / len(train_vectors)
            epoch_time = self.get_time_taken(epoch_start_time)

            print(
                "Epoch",
                epoch + 1,
                "Average loss:",
                round(average_loss, 4),
                "Time:",
                round(epoch_time, 4),
                "seconds"
            )

    # This function predicts one review as positive or negative.
    def predict_one(self, review):
        vector = self.review_to_vector(review)
        probability = self.predict_probability(vector)

        if probability >= 0.5:
            return 1
        else:
            return 0

    # This function predicts many reviews.
    def predict_many(self, reviews):
        predictions = []

        for review in reviews:
            predictions.append(self.predict_one(review))

        return predictions

    # This function prints the final evaluation results.
    def evaluate(self, true_labels, predictions):
        metrics = EvaluationMetrics()

        tp, fp, fn, tn = metrics.confusion_matrix(true_labels, predictions)

        print()
        print("Confusion Matrix")
        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print("TN:", tn)

        print()
        print("Accuracy:", round(metrics.accuracy(true_labels, predictions), 4))
        print("Precision:", round(metrics.precision(true_labels, predictions), 4))
        print("Recall:", round(metrics.recall(true_labels, predictions), 4))
        print("F1 Score:", round(metrics.f1_score(true_labels, predictions), 4))

    # This helper function changes 1 and 0 into readable labels.
    def label_to_text(self, label):
        if label == 1:
            return "Positive"
        else:
            return "Negative"

    # This helper function gives a simple explanation for model behavior.
    # It does not change the model prediction.
    # It is only used after prediction for error analysis,so it helps connect wrong predictions to possible model limitations.
    def explain_prediction(self, review, true_label, predicted_label):
        cleaned_review = self.clean_text(review)

        if true_label == predicted_label:
            return "Correct prediction. The model prediction matched the true sentiment."

        if "not " in cleaned_review or "n't" in cleaned_review:
            return "Possible negation issue. Bag-of-Words may not fully understand phrases like 'not good'."

        if "but" in cleaned_review or "however" in cleaned_review:
            return "Possible mixed-sentiment issue. The review may contain both positive and negative opinions."

        return "Possible issue: important sentiment words may be rare, missing from the vocabulary, or outweighed by other words."

    # This function prints detailed examples for error analysis.
    # error analysis which prints CSV rows 801 to 840.
    def error_analysis(self, test_reviews, test_labels, test_start_csv_row):
        print()
        print("Error Analysis")

        sample_count = min(self.error_analysis_limit, len(test_reviews))

        print("Examples analyzed:", sample_count)
        print("CSV rows analyzed:", test_start_csv_row, "to", test_start_csv_row + sample_count - 1)

        correct_count = 0
        wrong_count = 0

        for i in range(sample_count):
            review = test_reviews[i]
            true_label = test_labels[i]

            vector = self.review_to_vector(review)
            probability = self.predict_probability(vector)

            if probability >= 0.5:
                predicted_label = 1
            else:
                predicted_label = 0

            if true_label == predicted_label:
                result = "Correct"
                correct_count += 1
            else:
                result = "Wrong"
                wrong_count += 1

            explanation = self.explain_prediction(review, true_label, predicted_label)

            short_review = review

            if len(short_review) > 400:
                short_review = short_review[:400] + "..."

            print()
            print("Example", i + 1)
            print("CSV row:", test_start_csv_row + i)
            print("True label:", self.label_to_text(true_label))
            print("Predicted label:", self.label_to_text(predicted_label))
            print("Positive probability:", round(probability, 4))
            print("Result:", result)
            print("Explanation:", explanation)
            print("Review:", short_review)

        print()
        print("Error Analysis Summary")
        print("Correct examples:", correct_count)
        print("Wrong examples:", wrong_count)

        if sample_count > 0:
            print("Accuracy on these examples:", round(correct_count / sample_count, 4))

    # This function runs the whole project from start to finish.
    def run(self, filename):
        total_start_time = time.perf_counter()

        print("Loading data...")
        load_start_time = time.perf_counter()
        data = self.load_data(filename, self.data_limit)
        load_time = self.get_time_taken(load_start_time)

        train_data, test_data = self.split_data(data)

        train_reviews = []
        train_labels = []
        test_reviews = []
        test_labels = []

        for review, label in train_data:
            train_reviews.append(review)
            train_labels.append(label)

        for review, label in test_data:
            test_reviews.append(review)
            test_labels.append(label)

        test_start_csv_row = len(train_reviews) + 1

        print()
        print("Data Used")
        print("Reviews loaded:", len(data))
        print("Training reviews:", len(train_reviews))
        print("Test reviews:", len(test_reviews))
        print("Training CSV rows: 1 to", len(train_reviews))
        print("Testing CSV rows:", test_start_csv_row, "to", len(data))

        print()
        print("Building vocabulary...")
        vocabulary_start_time = time.perf_counter()
        self.build_vocabulary(train_reviews)
        vocabulary_time = self.get_time_taken(vocabulary_start_time)
        print("Vocabulary size:", len(self.vocabulary))

        print()
        print("Training model...")
        training_start_time = time.perf_counter()
        self.train_model(train_reviews, train_labels)
        training_time = self.get_time_taken(training_start_time)

        print()
        print("Predicting...")
        prediction_start_time = time.perf_counter()
        predictions = self.predict_many(test_reviews)
        prediction_time = self.get_time_taken(prediction_start_time)

        self.evaluate(test_labels, predictions)

        self.error_analysis(test_reviews, test_labels, test_start_csv_row)

        total_time = self.get_time_taken(total_start_time)

        print()
        print("Runtime Summary")
        print("Loading data:", round(load_time, 4), "seconds")
        print("Building vocabulary:", round(vocabulary_time, 4), "seconds")
        print("Training model:", round(training_time, 4), "seconds")
        print("Prediction/testing:", round(prediction_time, 4), "seconds")
        print("Total program time:", round(total_time, 4), "seconds")


# This part starts the program.
if __name__ == "__main__":
    dataset_file = "IMDB Dataset.csv"

    if len(sys.argv) > 1:
        dataset_file = sys.argv[1]

    project = SentimentAnalysis()
    project.run(dataset_file)