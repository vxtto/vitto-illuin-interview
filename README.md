# Information Retrieval with TF-IDF Matrix

### Content
The proposed implementation performs information retrieval on documents starting from a question.
The documents are aggregations of the paragraphs from each entry of the [SQuAD 1.1 dataset](https://deepai.org/dataset/squad).
A solution that matched questions to paragraphs was developed on a subset during the exploratory phase (code snippets in `src/testing_illuin.ipynb`), yet was hard to scale due to RAM constraints when running on the training set.

### Usage
1. Run the executable in `src/dist/`
2. Provide the path to the SQuAD JSON file of your choice (both training and testing work)
3. Wait a few seconds for the dataset to be preprocessed then select a number of questions you want to retrieve documents for
4. Repeat
