import pandas as pd
import re
import string
from collections import Counter
import math

df = pd.read_csv("IMDB Dataset.csv")


df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})


df = df.sample(1000, random_state=42).reset_index(drop=True)



def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text

df["clean_review"] = df["review"].apply(clean_text)


stop_words = {
    "the", "and", "is", "in", "to", "of", "a", "an", "it", "this",
    "that", "was", "for", "on", "with", "as", "but", "at", "by",
    "from", "or", "be", "are", "were", "has", "have", "had", "i",
    "you", "he", "she", "they", "we", "my", "your", "his", "her",
    "their", "our", "its"
}




def tokenize(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

df["tokens"] = df["clean_review"].apply(tokenize)

#print(df["tokens"][0][:30])

word_counts = Counter()

for tokens in df["tokens"]: 
    word_counts.update(tokens)

#print("Total unique words:", len(word_counts))
#print(word_counts.most_common(20))

VOCAB_SIZE = 3000

most_common_words = word_counts.most_common(VOCAB_SIZE)

vocab = {}

for i, (word, count) in enumerate(most_common_words):
    vocab[word] = i

#print("Vocabulary size:", len(vocab))
#print(list(vocab.items())[:10])

def vectorize(tokens):

    vector = [0] * len(vocab)

    for word in tokens:
        if word in vocab:
            index = vocab[word]
            vector[index] += 1

    return vector


X = []

for tokens in df["tokens"]:
    X.append(vectorize(tokens))

y = df["label"].tolist()

print("Vector length:", len(X[0]))
print(X[0][:30])
print("Label:", y[0])


split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


def euclidean_distance(vec1, vec2):
    distance = 0

    for i in range(len(vec1)):
        distance += (vec1[i] - vec2[i]) ** 2

    return math.sqrt(distance)


distance = euclidean_distance(X_train[0], X_train[1])
print(distance)


def predict_knn(test_vector, X_train, y_train, k=5):

    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(test_vector, X_train[i])
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])

    nearest_neighbors = distances[:k]

    labels = [label for distance, label in nearest_neighbors]

    prediction = max(set(labels), key=labels.count)

    return prediction


prediction = predict_knn(X_test[0], X_train, y_train, k=5)

print("Predicted:", prediction)
print("Actual:", y_test[0])

correct = 0

y_true = []
y_pred = []

for i in range(len(X_test)):
    prediction = predict_knn(X_test[i], X_train, y_train, k=5)

    y_true.append(y_test[i])
    y_pred.append(prediction)

    if prediction == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print("Correct predictions:", correct)
print("Total test reviews:", len(X_test))
print("Accuracy:", accuracy)

TP = 0
TN = 0
FP = 0
FN = 0


for actual, predicted in zip(y_true, y_pred):

    if actual == 1 and predicted == 1:
        TP += 1

    elif actual == 0 and predicted == 0:
        TN += 1

    elif actual == 0 and predicted == 1:
        FP += 1

    elif actual == 1 and predicted == 0:
        FN += 1

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)


accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP) if (TP + FP) != 0 else 0

recall = TP / (TP + FN) if (TP + FN) != 0 else 0

f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)









