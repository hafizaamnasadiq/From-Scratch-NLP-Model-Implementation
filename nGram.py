import math
from loadData import LoadData

class CharLevelNGram:
    def __init__(self, dataset):
        self.dataset = dataset
        self.ngram = []
        self.class_ngram_counts = {} # Stores n-gram frequencies for each class
        self.class_total_counts =  {} # Stores total number of n-grams per class
        self.vocabulary = set()  # Vocabulary of all unique n-grams
        self.class_priors = {}        # {label: probability}
        self.class_doc_counts = {}

    def extract_char_ngrams(self, n):
        result = []

        for text, label in self.dataset:
            print(text)
            # Add padding
            cleaned_text = text.lower().strip()
            padded_text = "_" * (n - 1) + cleaned_text + "_" * (n - 1)

            # Generate overlapping n-grams
            ngrams = [
                padded_text[i:i+n]
                for i in range(len(padded_text) - n + 1)
            ]
            result.append((ngrams, label))
            self.ngram.extend(ngrams)
            for ngram in ngrams:
                self.build_class_frequencies(ngram, label)

        return result

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

        
    def compute_class_priors(self): #laplace smoothing
        total_docs = len(dataset)
        
        # Count documents per class
        for text,label in self.dataset:
            if label not in self.class_doc_counts:
                self.class_doc_counts[label] = 0

            self.class_doc_counts[label] += 1
        # Compute prior probabilities
        self.class_priors = {}
        for label in self.class_doc_counts:
            self.class_priors[label] = self.class_doc_counts[label] / total_docs
       
        return self.class_priors

    def train_model(self,n):
        self.extract_char_ngrams(n)
        self.compute_class_priors();
        self.predict_text();

#    model = {
#     "priors": ...,
#     "class_ngram_counts": ...,
#     "class_total_counts": ...,
#     "vocabulary": ...,
#     "n": 3
# }
    
    def score_text(self):
        vocabulary_size = len(self.vocabulary)
        # Compute Score 
        scores = {}
        for label in self.class_priors:
            score = math.log(self.class_priors[label]) # class priority
            total_count = self.class_total_counts[label] # total class counts of n-grams
            # Now compute probability for each n-gram
            for ngram in self.ngram:
                count = self.class_ngram_counts[label].get(ngram, 0)
                # Laplace smoothing
                probability = (
                    (count + 1) /
                    (total_count + vocabulary_size)
                )
                score += math.log(probability)
        
            scores[label] = score
        print(len(self.class_priors))
        return scores
    
    def predict_text(self):
        scores = self.score_text();
        print("Scores\n")
        print(scores)
        return max(scores, key=scores.get)

    def predict_batch(model, texts):
        pass

    def evaluate_model(model, dataset):
        pass



dataset = [("free money now", 1),
          ("win money now", 1),        
          ("call me now", -1),         
          ("let's meet now", -1),

        ]
model = CharLevelNGram(dataset)

model.train_model(3)

# Train model
model.train_model(3)

# Print statistics
# print("\nN-gram Counts\n")
# print(model.class_ngram_counts)

# print("\nN-grams\n")
# print(model.ngram)

# print("\nTotal class counts\n")
# print(model.class_total_counts)

# print("\nVocabulary\n")
# print(model.vocabulary)

# print("\nClass priors\n")
# print(model.class_priors)


loader = LoadData()
df = loader.load_data()

print(df.head())