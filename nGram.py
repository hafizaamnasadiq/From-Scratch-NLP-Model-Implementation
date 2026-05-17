import math
import time
from loadData import LoadData
from evalMet import ConfusionMetrics

class CharLevelNGram:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n
        # self.ngram = []
        self.class_ngram_counts = {} # Stores n-gram frequencies for each class
        self.class_total_counts =  {} # Stores total number of n-grams per class
        self.vocabulary = set()  # Vocabulary of all unique n-grams
        self.class_priors = {}        # {label: probability}
        self.class_doc_counts = {}

    def reset_model(self):
        self.class_ngram_counts = {}
        self.class_total_counts = {}
        self.vocabulary = set()
        self.class_doc_counts = {}
        self.class_priors = {}

    def extract_char_ngrams(self, text):

        for text, label in self.dataset:
            # Add padding(_)
            padded_text = "_" * (self.n - 1) + text + "_" * (self.n - 1)

            # Generate overlapping n-grams
            ngrams = [
                padded_text[i:i+self.n]
                for i in range(len(padded_text) - self.n + 1)
            ]

        return ngrams

    def build_class_frequencies(self, ngram, label):
         # Initialize class dictionaries if needed
        if label not in self.class_ngram_counts:
            self.class_ngram_counts[label] = {}

        if label not in self.class_total_counts:
            self.class_total_counts[label] = 0

        self.vocabulary.add(ngram)
        # Initialize count if first occurrence
        if ngram not in self.class_ngram_counts[label]:
            self.class_ngram_counts[label][ngram] = 0

        # Increment frequency
        self.class_ngram_counts[label][ngram] += 1

        # Increment total count for this class
        self.class_total_counts[label] += 1

        
    def compute_class_priors(self):
        total_docs = len(self.dataset) # total rows of data
        
        self.class_doc_counts = {}
        for text, label in self.dataset:
            self.class_doc_counts[label] = self.class_doc_counts.get(label, 0) + 1
        
        self.class_priors = {}
        for label in self.class_doc_counts:
            self.class_priors[label] = self.class_doc_counts[label] / total_docs

        return self.class_priors

    def train_model(self, epochs=1):
        for epoch in range(epochs):
            start_time = time.time()

            for text, label in self.dataset:
                ngrams = self.extract_char_ngrams(text)

                for ngram in ngrams:
                    self.build_class_frequencies(ngram, label)

            self.compute_class_priors();
            end_time = time.time()
            print(f"Training time Epoch {epoch+1}/{epochs} - ", end_time - start_time, "seconds")


    def score_text(self,text):
        ngrams = self.extract_char_ngrams(text)
        vocabulary_size = len(self.vocabulary)
        # Compute Score 
        scores = {}

        for label in self.class_priors:
            score = math.log(self.class_priors[label]) # class priority
            total_count = self.class_total_counts[label] # total class counts of n-grams
            # Now compute probability for each n-gram

            for ngram in ngrams:
                count = self.class_ngram_counts[label].get(ngram, 0)
                # Laplace smoothing
                probability = (
                    (count + 1) /
                    (total_count + vocabulary_size)
                )
                score += math.log(probability)
        
            scores[label] = score
        return scores
    
    # Predict single text
    def predict_text(self,text):
        scores = self.score_text(text);
        prediction = max(scores, key=scores.get)
        return prediction

    def predict_batch(self, texts):
        predictions = []

        for text in texts:
            prediction = self.predict_text(text)
            predictions.append(prediction)
        return predictions

    def test_model(self, dataset):
        correct = 0
        total = len(dataset)
        y_true = []
        y_pred = []

        for text, true_label in dataset:
            predicted = self.predict_text(text)
            y_true.append(true_label)
            y_pred.append(predicted)
            if predicted == true_label:
                correct += 1
        return y_true, y_pred



# print("STATIC DATA")
# dataset = [("free money now", 0),
#           ("win money now", 0),        
#           ("call me now", 1),         
#           ("let's meet now", 1),

#         ]

# model = CharLevelNGram(dataset,n=4)
# model.train_model()

# print(model.predict_text("free win money now"))

# print("BATCH TESTING")
# texts = [
#     ("free money offer",0),
#     ("let's meet today",1),
#     ("win cash now",0),
#     ("call me later",1),
#     ("call me urgent",0)
# ]

# print(model.test_model(texts))
# print("PRIORS:", model.class_priors)
# print("TOTAL COUNTS:", model.class_total_counts)

print("---------------------TRAINING")
loader = LoadData()
df = loader.load_data()

train_df = df.head(800) #Training 800
test_df = df.tail(200) # Testing 200
#TRAINING
# label , clean_review
dataset = list(zip(train_df["clean_review"], train_df["label"]))

model = CharLevelNGram(dataset, n=4)
model.train_model()

#Testing
print("------------------------TESTING")
testdataset = list(zip(test_df["clean_review"], test_df["label"]))
y_true,y_pred = model.test_model(testdataset)
print("PRIORS:", model.class_priors)
print("Total Class Counts:", model.class_total_counts) 


# Evaluation
# Initialize the reused class
metrics = ConfusionMetrics(y_true, y_pred)

print(f"Accuracy: {metrics.accuracy()}")
print(f"Recall: {metrics.recall()}")
print(f"Precision: {metrics.precision()}")
print(f"F1 Score: {metrics.f1_score()}")


# Print statistics
# print("\nN-gram Counts\n")
# print(model.class_ngram_counts)

# print("\nN-grams\n")
# print(model.ngram)


# print("\nVocabulary\n")
# print(model.vocabulary)