# CNN Text Classification with WordPiece Tokenization
This repository is a boilerplate of CNN model for single-label text classification task, built using PyTorch. The boilerplate will preprocess raw dataset (supported format: csv, json, pkl) and feed them to CNN model. Combinations of different text cleaning and tokenization methods are applied to dataset.

### Text cleaning methods
- [Alphabet Only] - This method removes stop words, retains Latin alphabets and space.
- [Special] - This method removes special characters, rarely used punctuation, numbers, and stop words.
- [Special Lowercase] - The same as [Special] except all letters are lowercase.

### Tokenization Methods
- [WordPiece] - WordPiece is a sub-word tokenization algorithm developed by Google. It is presented in "Japanese and Korean voice search" paper (Schuster and Nakajima, 2012). I use WordPiece tokenizer from BERT original implementation (Devlin et al., 2019) and developed my own tokenizer using News Landscape dataset (Horne et al., 2018).
- [Simple Word Tokenization] - Tokenization method with space as separator between each element.

### Embeddings
- [FastText] - Word vector representations developed by Facebook. For more information, please refer to "Advances in Pre-Training Distributed Word Representations" (Mikolov et al., 2018).
- [Trainable Embeddings] - Normal trainable embeddings provided by PyTorch.

### Model
CNN model in this repository is inspired by the architecture of CNN as used in "Convolutional Neural Network for Sentence Classification" paper (Kim, 2014).

### Usage
For example, You want to use [IMDB movie review dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Save the dataset in this directory:

```sh
dataset/
    movie-review/
        dataset.csv
```

and then, at [helpers.py](https://github.com/muhd-fadilah/cnn-text-classification-with-wordpiece-tokenization/blob/main/src/helpers.py), add this key and value of dictionary in ___get_classification_dataset_info_ of _DatasetHelper_ class:
```sh
 # Folder name
 "movie-review": {
                # File name
                "file_name": "dataset.csv", 
                # Support up to two columns (title and text)
                "used_columns": ["text"],
                # Title column is not used, so we set this to 'None' 
                "title_column": None,
                # Text column is named 'review' in example dataset
                "text_column": "review",
                # Label column is named 'sentiment' in example dataset
                "label_column": "sentiment", 
                # We set these to 0 since we don't use title column
                "title_max_length": {
                    "pre_trained_wordpiece_cased": 0,
                    "pre_trained_wordpiece_uncased": 0,
                    "self_trained_wordpiece_cased": 0,
                    "self_trained_wordpiece_uncased": 0,
                    "word_tokenize": 0,
                },
                # Set these to the value you wish
                "text_max_length": {
                    "pre_trained_wordpiece_cased": 500,
                    "pre_trained_wordpiece_uncased": 500,
                    "self_trained_wordpiece_cased": 500,
                    "self_trained_wordpiece_uncased": 500,
                    "word_tokenize": 500,
                },
            },
```