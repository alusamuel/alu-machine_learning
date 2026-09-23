# Word Embeddings

This directory contains exercises that turn text into numeric vectors, from
simple count-based encodings to trained embedding models.

## Files

- `0-bag_of_words.py`: creates a bag-of-words embedding matrix.
- `1-tf_idf.py`: creates a TF-IDF embedding matrix.
- `2-word2vec.py`: creates and trains a gensim `Word2Vec` model.
- `3-gensim_to_keras.py`: converts a trained gensim word2vec model into a
  trainable Keras `Embedding` layer.
- `4-fasttext.py`: creates and trains a gensim `FastText` model.
- `5-elmo`: answers the ELMo training question.

## Notes

- In both gensim models, `cbow=True` trains with CBOW (`sg=0`) and
  `cbow=False` trains with Skip-gram (`sg=1`).
- These files use the gensim 3.x API (`size`, `iter`, `get_keras_embedding`)
  and `get_feature_names` from older scikit-learn releases.
