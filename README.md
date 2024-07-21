# Learning Agency Lab Automated Essay Scoring 2.0

_A Kaggle competition to automate the ranking/scoring of essays on a scale of 1 to 6._

## Problem Description / Objective

This is a 'multi-class' classification problem where a given essay/observation can belong to 1, and only 1, out of 6 classes.

## Data Description

This section outlines the details of the datasets and data manipulations applied to them _(Source: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data)_.

### Data Description:

**Train data**: This dataset contains 17,307 sample essays with their corresponding labels _(1 to 6)_. The file contains 3 fields; 'essay_id' _(a unique identifier of the essay/observation)_, 'full_text' _(containing the full text of the essay)_, and 'score' _(the essay rank/class label)_. NB: The final training data contained 80 to 90% of the original as explained under **Validation data**. 

**Test / Submission data**: This dataset contains 3 sample essays that are to be ranked.

**Validation data**: This dataset contains 10%-20% of the **Train data** and has been derived using the following code:

    from sklearn.model_selection import StratifiedShuffleSplit as sss

    splits = sss(n_splits=1, test_size=0.1, random_state=42)

    for train_index, test_index in splits.split(df_train['full_text'], df_train['score']):
        X_train, X_test = df_train['full_text'][train_index], df_train['full_text'][test_index]
        y_train, y_test = df_train['score'][train_index], df_train['score'][test_index]

**External data**: GLoVe word embeddings sourced from _https://nlp.stanford.edu/projects/glove/_ were used to augment the train data. _(File name: glove.6B.300d)_

### Data Preprocessing:

Various text data preprocessing techniques were applied to the train dataset as described below.

#### 1. Data Augmentation using Word Embeddings: In order to increase the variety of words and the training sample size, the essays were augmented with synonyms calculated, stored and applied as a .pkl file using the code below:

    from gensim.models import KeyedVectors
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    glove_file = 'glove.6B.300d.txt'
    
    print("Loading word vectors")
    load_start = datetime.now()
    
    word_vectors = KeyedVectors.load_word2vec_format(fname=glove_file, binary=False, unicode_errors='ignore', no_header=True, limit=400000)
    
    load_end = datetime.now()
    print(f"Word vectors loaded in {load_end - load_start}\n")    
    
    def compute_most_similar(word):
        return word, word_vectors.most_similar(word, topn=1)[0][0]
    
    print("Precomputing similar words\n")
    comp_start = datetime.now()
    
    similar_words_dict = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_most_similar, word): word for word in word_vectors.index_to_key}
        for future in as_completed(futures):
            word, most_similar = future.result()
            similar_words_dict[word] = most_similar
    
    comp_end = datetime.now()
    print(f"Precompute complete in {comp_end - comp_start}")
    
    import pickle
    
    with open('similar_words_dict.pkl', 'wb') as fp:
        pickle.dump(similar_words_dict, fp)
        print('Dictionary saved successfully to file')
    
    with open('similar_words_dict.pkl', 'rb') as fp:
        similar_words_dict = pickle.load(fp)
        print('similar_words_dict loaded')
        
    from multiprocessing import Pool
    
    def augment_text_with_glove(text):
        augmented_text = []
        for word in text.split():
            if word in similar_words_dict:
                augmented_text.append(similar_words_dict[word])
            else:
                augmented_text.append(word)
        return ' '.join(augmented_text)
    
    def parallel_augment_texts(texts, num_workers=4):
        with Pool(num_workers) as pool:
            augmented_texts = list(pool.imap(augment_text_with_glove, texts))
        return augmented_texts
    
    print("Starting data augmentation now\n")
    aug_start = datetime.now()
    print("Augmentation start time is: ", aug_start)
    
    df_aug['full_text'] = parallel_augment_texts(df_train['full_text'].tolist())
    
    aug_end = datetime.now()
    print("\nAugmentation end time is: ", aug_end)
    print("\nData augmentation complete")


#### 2. Lemmatization and Stop Word Removal: For better understanding, noise reduction, processing efficiency, and relevance; words were reduced to their root form _(i.e. lemma or the dictionary form in the context of the surrounding text)_ and words with little or no value _(i.e. stop words such as 'the', 'is', etc.)_ were removed from the train word corpus using the code below:

    import spacy
    from concurrent.futures import ProcessPoolExecutor
    
    nlp = spacy.load("en_core_web_sm")
    
    # Function to apply lemmatization
    def lemmatize_text(text):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])
    
    def process_chunk(chunk):
        return chunk.apply(lemmatize_text)
    
    def parallel_lemmatize(data, num_processes):
        chunks = np.array_split(data, num_processes)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(process_chunk, chunks))
        return pd.concat(results, ignore_index=True)
    
    print("\nApplying lemmatization now")
    lem_start = datetime.now()
    print("\nLemmatization start time is: ", lem_start)
    
    num_processes = 8
    X_train = parallel_lemmatize(X_train, num_processes)
    
    lem_end = datetime.now()
    print("\nLemmatization end time is: ", lem_end)
    print("\nData lemmatization complete")

### Exploratory Data Analysis (EDA): TBC

## Modeling Approach

The modeling phase was iterative, whereby, a variety of models were built, with alterations ranging from the use of GANs to plain vanilla Text Classification models from the Tensorflow Sequential library. Here's a draft summary of the data preparation approaches applied and the 
libraries used to build the models.

### Model Selection: TBC

### Model Architecture: TBC

### Hyperparameter Tuning: TBC

### Training Procedure: TBC
  

## Evaluation Metrics: TBC

### Chosen Metrics: TBC

### Rationale: TBC

### Results: TBC

## Model Performance: TBC

### Validation Results: TBC

### Cross-Validation: TBC

### Error Analysis: TBC

## Conclusion: TBC

### Summary: TBC

### Challenges: TBC

### Future Work: TBC

## Appendices: TBC

### Additional Resources: TBC

## References: TBC

### Citations: TBC


