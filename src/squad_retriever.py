import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def load_dataframe():
    """
    Requests the path to the SQuAD formatted JSON.
    Loads and returns the data.
    :return: pandas DataFrame containing the data.
    """
    path = input("Insert the path to the SQuAD JSON:\n")
    return pd.read_json(path)


def retrieve_squad_data(df):
    """
    Extracts full documents, questions, and corresponding labels
    from the given dataframe.
    :param df: pandas dataframe formatted according to the SQuAD
    standard.
    :return: List of documents (each document is created by joining
    all paragraphs from a dataframe entry), list of questions, and
    a list of labels, one for each question, representing the document
    associated to each question.
    """
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
    """
    Computes the TF-IDF matrix from the given documents.
    :param docs: List of documents as strings.
    :return: A vectorizer that allows vectorization of
    any string based on the TF-IDF vocabulary.
    The TF-IDF matrix from the documents.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


def sample_qs(questions, labels, seed=26):
    """
    Utility function implemented to subsample a set
    of questions.
    :param questions: list containing all questions from
    the dataset.
    :param labels: list containing the document labels
    associated to each question.
    :param seed: randomness seed, default 26.
    :return: Two numpy arrays of the same length. The first
    containing questions and the second containing labels.
    """
    num_qs = len(questions)
    # Ask the user how many questions he will test the script on
    # No checks are performed on the max number. The script can crash
    # if max is exceeded.
    sample_size = input(f"Insert number of questions to retrieve (max {num_qs}):")
    questions = np.asarray(questions)
    labels = np.asarray(labels)
    # Create an array with all indices.
    indices = np.arange(num_qs)
    # Shuffle the array. This is done so when the first N
    # indices are taken, they correspond to random questions.
    np.random.seed(seed)
    np.random.shuffle(indices)
    sample_indices = indices[:int(sample_size)]
    return questions[sample_indices], labels[sample_indices]


def retrieve_from_questions(questions, vectorizer, docs_tfidf):
    """
    Retrieves the documents that appear to contain the answers to the
    given questions.
    :param questions: Set of questions for which documents containing answers
    will be searched.
    :param vectorizer: Vectorizer fit on the documents.
    :param docs_tfidf: TF-IDF matrix of documents.
    :return: Predicted documents.
    """
    # All questions are vectorized so they can be compared to the doc
    # vectors.
    vectorized_qs = vectorizer.transform(questions)
    # Compute the cosine similarities between each question and all documents.
    # This generates a QxD matrix, where Q is the number of questions and D the
    # number of documents. For each row, the index of the column containing the
    # highest value corresponds to the document most similar to the question.
    # This implementation assumes that the text with the highest similarity will be
    # the one that is able to answer the question.
    # Because of this the QxD matrix is squeezed into a Qx1 matrix where each entry
    # corresponds to the index of the max element of the matching row.
    retrieved_labels = csr_matrix.argmax(
        cosine_similarity(vectorized_qs, docs_tfidf, dense_output=False), axis=1
    )
    return retrieved_labels


def evaluate(l_predicted, l):
    """
    Evaluates the algorithm with its accuracy.
    :param l_predicted: Labels predicted by the implementation.
    :param l: Ground truth.
    """
    sample_size = len(l)
    # Counts the number of matching predictions
    num_correct = np.count_nonzero(np.equal(l_predicted, l.reshape(-1, 1)))
    accuracy = num_correct / sample_size * 100
    print(f"The algorithm retrieved {num_correct} associated texts correctly."
          f"\nOut of {sample_size}, that is a {accuracy:.2f}% accuracy.")


if __name__ == '__main__':
    print("Disclaimer: for computational capacity purposes the task has been simplified."
          "The granularity of the documents to be retrieved has been made larger by "
          "aggregating all paragraphs of SQuAD entry into a document. Hence, the end result"
          "of the algorithm associates each question to its corresponding aggregate of paragraphs.")
    while True:
        try:
            # The algorithm pipeline proceeds as follows:
            # 1. Load the SQuAD data
            squad_df = load_dataframe()
            # 2. Massage the data to extract documents (created by
            #   merging all paragraphs from one entry), questions,
            #   and labels
            documents, questions, labels = retrieve_squad_data(squad_df)
            # 3. Fit a TF-IDF matrix on the set of documents.
            vectorizer, docs_tfidf = fit_tfidf(documents)
            # 4. Subsample the questions out of the total number of questions.
            #   This is done to not have memory errors during the computation of
            #   similarities.
            questions_sample, labels_sample = sample_qs(questions, labels)
            # 5. Retrieve the documents that seem to answer the sample of
            #   questions.
            retrieved_labels = retrieve_from_questions(questions_sample, vectorizer, docs_tfidf)
            # 6. Evaluate the goodness of the result.
            evaluate(retrieved_labels, labels_sample)
        except KeyboardInterrupt:
            exit()
        except ValueError:
            print("Invalid path")