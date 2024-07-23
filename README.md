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

#### 1. Data Augmentation using Word Embeddings: In order to increase the variety of words and the training sample size, the train data was augmented with synonyms calculated, stored and applied as a .pkl file using the code below:

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

#### 3. Text Vectorization _(Implemented in the final phase for some models)_: To support experiments where the 'stacked' models approach _(to improve prediction quality)_ was used; the text corpus was vectorized using the 'TfidfVectorizer' library from 'sklearn.feature_extraction.text' was used as shown below: 

    tfidf_vectorizer = TfidfVectorizer(max_features=1000, analyzer='word', ngram_range=(1,9))
    X_train = tfidf_vectorizer.fit_transform(X_train).toarray()
    X_test = tfidf_vectorizer.transform(X_test).toarray()

#### 4. Tokenization and Padding: Since the final model, including in the 'stacked' approach; was a TensorFlow Sequential neural network model, data was _(i.e. in addition to synonym augmentation and lemmatization described above)_ tokenized using B.P.E. _(Byte Pair Encoding)_ approach and subsequently padded to ensure that the sequences fed to the model were of the same length. The 'padded_texts' were then converted to TensorFlow Datasets. The code for each of these steps is shown below:

    # Tokenzing data for neural network
    
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=400000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(X_train_nn, trainer)
    tokenizer.save("tokenizer.json")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenized_texts = [tokenizer.encode(text).ids for text in X_train_nn]

    # Pad the tokenized texts
    
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded_texts = pad_sequences(tokenized_texts, maxlen=max_length, padding='post', truncating='post')

    # Create a TensorFlow dataset
    
    dataset = tf.data.Dataset.from_tensor_slices((padded_texts, y_train_nn))
    dataset = dataset.shuffle(len(padded_texts), seed=SEED).batch(128)

### Exploratory Data Analysis (EDA): 

Analysis of the training dataset revealed a class imbalance _(although, from a real world perspective, it makes sense that the bulk of the essays would likely receive a rank closer to the middle, this could potentially bias the model, and hence was considered as an imbalance and treated using 'Synthetic Minority Oversampling Technique S.M.O.T.E.' {in some model versions but not the final}.)_, whereby, a majority of the essays had a rank/score of 3. The code snippet to observe this and the counts by rank / category are shown below:
    
    df_train['score'].value_counts()

    |Score|Record Count|
    |-----|------------|
    |1    |1252        |
    |-----|------------|
    |2    |4723        |
    |-----|------------|    
    |3    |6280        |
    |-----|------------|
    |4    |3926        |
    |-----|------------|
    |5    |970         |
    |-----|------------|
    |6    |156         |
    |-----|------------|
    

## Modeling Approach

This section describes the modeling approach including - model selection, architecture, hyperparameter tuning and the training procedure for the models.

### Model Selection: 

Since this was a multi-class text classification problem, the models selected for experimentation were different architectural configurations of Neural Network models from TensorFlows' Sequential library with a 'softmax' activated 'Dense' layer _(containing 6 units)_ as the final layer.

### Model Architecture: 

Here is an example of a simple architecture of one of the text classification models:

    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_length,)),
    tf.keras.layers.Embedding(input_dim=400000, output_dim=128), # input dim increased 30/05/2024
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)), # added 28/05/2024
    tf.keras.layers.Dropout(0.4), # added 28/05/2024   
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),    
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(6, activation='softmax')
    ])

### Hyperparameter Tuning: 

### Training Procedure: 
  

## Evaluation Metrics: 

### Chosen Metrics: 

### Rationale: 

### Results: 

## Model Performance: 

### Validation Results: 

### Cross-Validation: 

### Error Analysis: 

## Conclusion: 

### Summary: 

### Challenges: 

### Future Work: 

## Appendices: 

### Additional Resources: 

## References: 

### Citations: 


