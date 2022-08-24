import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def load_dataframe():
    path = input("Insert the path to the SQuAD JSON:\n")
    return pd.read_json(path)


def retrieve_squad_data(df):
    def process_paragraph(paragraph):
        return ' '.join(preprocess_string(paragraph))

    docs, questions, labels = [], [], []

    for entry in df.iterrows():
        # Variable that will host all paragraphs from an entry
        paragraphs = []
        # Iterate through all paragraphs
        for paragraph in entry[1]['data']['paragraphs']:
            # Remove stop words and concatenate to form the document
            processed_paragraph = process_paragraph(paragraph['context'])
            paragraphs.append(processed_paragraph)
            # Save all questions
            qas = paragraph['qas']
            # Append all questions from the current paragraph
            questions.extend(qa['question'] for qa in qas)
            # Append an equal number of labels to the labels structure
            labels.extend([entry[0]] * len(qas))
        docs.append(' '.join(paragraphs))

    return docs, questions, labels


def fit_tfidf(docs):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


def sample_qs(questions, labels, seed=26):
    num_qs = len(questions)
    sample_size = input(f"Insert number of questions to retrieve (max {num_qs}):")
    questions = np.asarray(questions)
    labels = np.asarray(labels)
    indices = np.arange(num_qs)
    np.random.seed(seed)
    np.random.shuffle(indices)
    sample_indices = indices[:int(sample_size)]
    return questions[sample_indices], labels[sample_indices]


def retrieve_from_questions(questions, vectorizer, docs_tfidf):
    vectorized_qs = vectorizer.transform(questions)
    retrieved_labels = csr_matrix.argmax(cosine_similarity(vectorized_qs, docs_tfidf, dense_output=False), axis=1)
    return retrieved_labels


def evaluate(l_predicted, l):
    sample_size = len(l)
    num_correct = np.count_nonzero(np.equal(l_predicted, l.reshape(-1, 1)))
    accuracy = num_correct / sample_size * 100
    print(f"The algorithm retrieved {num_correct} associated texts correctly."
          f"\nOut of {sample_size}, that is a {accuracy}% accuracy.")


if __name__ == '__main__':
    squad_df = load_dataframe()

    documents, questions, labels = retrieve_squad_data(squad_df)

    vectorizer, docs_tfidf = fit_tfidf(documents)

    questions_sample, labels_sample = sample_qs(questions, labels)

    retrieved_labels = retrieve_from_questions(questions_sample, vectorizer, docs_tfidf)

    evaluate(retrieved_labels, labels_sample)
