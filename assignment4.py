import os
import math
# Name this file assignment4.py when you submit

class bag_of_words_model:

  def __init__(self, directory):
    # directory is the full path to a directory containing trials through state space
    self.vocabulary = []
    self.term_to_index = {}
    self.idf_vector = []
    self.num_docs = 0
    # Return nothing

    if not os.path.isdir(directory):
      print("Input directory not found")
      return
    
    df_counts = {}
    # Pass through each file and do DF calculation 
    for file in os.listdir(directory): 
      file_path = os.path.join(directory, file)

      with open(file_path, 'r', encoding="utf-8") as f: 
        words = f.read().split()
      unique_words = set(words)
      
      for word in unique_words:
        df_counts[word] = df_counts.get(word, 0) + 1
      
      self.num_docs += 1
    
    ### Build dictionary and reverse dictionary to get words
    self.vocabulary = sorted(df_counts.keys())
    for i, term in enumerate (self.vocabulary):
      self.term_to_index[term] = i

    ### build IDF vector
    for term in self.vocabulary: 
      df = df_counts[term]
      idf_value = math.log(self.num_docs / df, 2)
      self.idf_vector.append(idf_value)

  def tf_idf(self, document_filepath):
    # document_filepath is the full file path to a test document
    with open(document_filepath, 'r', encoding="utf-8") as f: 
      text = f.read()
    words = text.split()

    tf_counts = [0.0] * len(self.vocabulary)

    for word in words: 
        word_index = self.term_to_index.get(word)
        if word_index is not None: 
          tf_counts[word_index] += 1.0
    
    doc_length = len(words)
    if doc_length > 0:
      tf_counts = [count / doc_length for count in tf_counts]

    tf_idf_vector = []
    for i in range(len(self.vocabulary)): 
      value = tf_counts[i] * self.idf_vector[i]
      tf_idf_vector.append(value)
    
    # Return the term frequency-inverse document frequency vector for the document
    return tf_idf_vector


  def predict(self, document_filepath, business_weights, entertainment_weights, politics_weights):
    # document_filepath is the full file path to a test document
    # business_weights is a list of weights for the business artificial neuron
    # entertainment_weights is a list of weights for the entertainment artificial neuron
    # politics_weights is a list of weights for the politics artificial neuron
    input_tf_idf = self.tf_idf(document_filepath)

    # Random score to prevent error
    if not input_tf_idf: 
      scores = [1/3, 1/3, 1/3]
      return 'unknown', scores
    
    # compute result
    business_y = 0.0
    entertainment_y = 0.0
    politics_y = 0.0

    for i in range(len(input_tf_idf)): 
      business_y += business_weights[i] * input_tf_idf[i]
      entertainment_y += entertainment_weights[i] * input_tf_idf[i]
      politics_y += politics_weights[i] * input_tf_idf[i]
    
    ### Softmax activiation
    raw_scores = [business_y, entertainment_y, politics_y]
    exp_scores = []
    
    max_raw = max(raw_scores)
    exp_scores = [math.exp(s - max_raw) for s in raw_scores]
    sum_exp = sum(exp_scores)
    scores = [es /sum_exp for es in exp_scores]

    topics = ['business', 'entertainment', 'politics']
    max_index = 0
    for i in range(1, len(scores)): 
      if scores[i] > scores[max_index]: 
        max_index = i
    
    predicted_label = topics[max_index]

    # Return the predicted label from the neural network model
    # Return the score from each neuron
    return predicted_label, scores

def read_weights(path):
  with open(path, "r", encoding="utf-8") as f:
    text = f.read().strip()

  # make commas behave like spaces
  text = text.replace(",", " ")

  parts = text.split()
  weights = []
  for p in parts:
    weights.append(float(p))
  return weights

if __name__ == "__main__":
  print("Hello world")
  example_dir = "/Users/tungtranthanh/Document_local/Documents/Fall_2025/COMP_3106/A4/Examples/Example1"

  training_dir = os.path.join(example_dir, "training_documents")
  model = bag_of_words_model(training_dir)

  test_doc_path = os.path.join(example_dir, "test_document.txt")
  business_weights_path = os.path.join(example_dir, "business_weights.txt")
  entertainment_weights_path = os.path.join(example_dir, "entertainment_weights.txt")
  politics_weights_path = os.path.join(example_dir, "politics_weights.txt")

  business_weights = read_weights(business_weights_path)
  entertainment_weights = read_weights(entertainment_weights_path)
  politics_weights = read_weights(politics_weights_path)

  tf_idf_vec = model.tf_idf(test_doc_path)
  print("TF-IDF vector:")
  print(tf_idf_vec)

  predicted_label, scores = model.predict(
    test_doc_path,
    business_weights,
    entertainment_weights,
    politics_weights
  )

  print("Predicted label:", predicted_label)
  print("Scores:", scores)