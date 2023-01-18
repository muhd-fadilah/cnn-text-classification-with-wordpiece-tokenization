import os
import re
import string
import io
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer

class TextCleaner:
    def __init__(self, unused_punctuation: str, stop_words: set, text_cleaning_types: dict) -> None:
        #unused punctuation
        self.unused_punctuation = unused_punctuation
        
        #stop words
        self.stop_words = stop_words

        #text cleaning types
        self.text_cleaning_types = text_cleaning_types
        
    def __remove_special(self, text: str):
        #remove tabs, new lines, and unicode identifier
        text = text.replace("\\t", " ").replace("\\n", " ")

        #decode unicode and ignore any error
        return text.encode('ascii', 'ignore').decode()

    def __remove_number(self, text: str):
        #remove any number in given regex format
        return re.sub(r"[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)*", " ", text)

    def __remove_punctuation(self, text: str):
        #remove punctuation based on given list
        return text.translate(str.maketrans(self.unused_punctuation, " " * len(self.unused_punctuation)))

    def __remove_multiple_spaces(self, text: str):
        #split text into words
        elements = text.split()
        result = ''

        all_punctuation = string.punctuation
        used_punctuation = ''

        #define punctuation used in text
        for punctuation in all_punctuation:
            if punctuation not in self.unused_punctuation:
                used_punctuation = used_punctuation + punctuation

        #iterate over words
        for element in elements:
            #give space if current element is word
            if element not in used_punctuation:
                result = result + " " + element
                continue

            #don't give space if current element is punctuation
            result = result + element

        return result

    def __remove_stopwords(self, text: str):
        text = text.split()

        #remove stop words based on defined stop words set
        return ' '.join(word for word in text if word.lower() not in self.stop_words)

    def alphabet_only(self, text: str):
        #convert text into lowercase
        text = text.lower()

        #tokenize text into words
        words = word_tokenize(text)

        words = [word for word in words if (word.isalpha() and word not in self.stop_words)]

        #concat remaining words
        return " ".join(words)

    def special(self, text: str, lowercase: bool = False):
        #preprocessing steps
        special_removed = self.__remove_special(text)
        number_removed = self.__remove_number(special_removed.lower() if lowercase else special_removed)
        punctuation_removed = self.__remove_punctuation(number_removed)
        stopwords_removed = self.__remove_stopwords(punctuation_removed)
        multiple_space_removed = self.__remove_multiple_spaces(stopwords_removed)

        return multiple_space_removed

    def txt_file(self, text: str, txt_instance: io.TextIOWrapper):
        #preprocessing steps
        special_removed = self.__remove_special(text)
        number_removed = self.__remove_number(special_removed)
        punctuation_removed = self.__remove_punctuation(number_removed)
        stopwords_removed = self.__remove_stopwords(punctuation_removed)
        multiple_space_removed = self.__remove_multiple_spaces(stopwords_removed)
        
        #write to .txt file
        txt_instance.write(multiple_space_removed + "\n")

    def classification_dataset(self, dataframe: pd.DataFrame, preprocessing_result: dict, origin_column: str, result_column: str):
        for text_cleaning_type in self.text_cleaning_types.keys():
            preprocessing_result[f"{result_column}_{text_cleaning_type}"] = dataframe[origin_column].apply(
                self.text_cleaning_types[text_cleaning_type]["method"]
            )
        
        return preprocessing_result

class PreprocessingHelper:
    def __init__(self) -> None:
        #unused punctuation
        self.unused_punctuation = "#$%+<=>@[\\]^_{|}~"

        #stop words
        self.stop_words = self.__get_stopwords()

        #unused punctuation
        self.unused_punctuation = "#$%+<=>@[\\]^_{|}~"

        #text cleaning types
        self.text_cleaning_types = self.__get_text_cleaning_types()

        #text cleaner instance
        self.text_cleaner = TextCleaner(
            unused_punctuation=self.unused_punctuation, 
            stop_words=self.stop_words,
            text_cleaning_types=self.text_cleaning_types
        )

    def __get_stopwords(self):
        #nltk stop words
        stop_words = set(stopwords.words('english'))

        #exclude some default nltk stop words
        not_stopwords = {
            'no', 'nor', 'not', 
            'don', "don't", 'ain', 
            'aren', "aren't", 'couldn', 
            "couldn't", 'didn', "didn't", 
            'doesn', "doesn't", 'hadn',
            "hadn't", 'hasn', "hasn't",
            'haven', "haven't", 'isn', 
            "isn't", 'ma', 'mightn', 
            "mightn't", 'mustn', "mustn't", 
            'needn', "needn't", 'shan',
            "shan't", 'shouldn', "shouldn't", 
            'wasn', "wasn't", 'weren', 
            "weren't", 'won', "won't", 
            'wouldn', "wouldn't"
        }

        #final stop words
        return set([word for word in stop_words if word not in not_stopwords])

    def __get_text_cleaning_types(self):
        return {
            "alphabet_only": {
                "tokenizer_types": ["word_tokenize", "word_tokenize_fasttext", "self_trained_wordpiece_uncased", "pre_trained_wordpiece_uncased"],
                "method": lambda text: self.text_cleaner.alphabet_only(text),
            },
            "special": {
                "tokenizer_types": ["word_tokenize", "word_tokenize_fasttext", "self_trained_wordpiece_cased", "pre_trained_wordpiece_cased"],
                "method": lambda text: self.text_cleaner.special(text),
            },
            "special_lowercase": {
                "tokenizer_types": ["word_tokenize", "word_tokenize_fasttext", "self_trained_wordpiece_uncased", "pre_trained_wordpiece_uncased"],
                "method": lambda text: self.text_cleaner.special(text),
            }, 
        }

    def get_tokenizer_types(self):
        #wordpiece tokenizers
        self.self_trained_wordpiece_uncased = BertTokenizer.from_pretrained(os.path.join(os.getcwd(), "wordpiece", "uncased"))
        self.self_trained_wordpiece_cased = BertTokenizer.from_pretrained(os.path.join(os.getcwd(), "wordpiece", "cased"))
        self.bert_base_uncased = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_base_cased = BertTokenizer.from_pretrained("bert-base-cased")

        return  {
            "pre_trained_wordpiece_cased": {
                "tokenizer": lambda text: self.bert_base_cased.encode(text, add_special_tokens=False),
            },
            "pre_trained_wordpiece_uncased": {
                "tokenizer": lambda text: self.bert_base_uncased.encode(text, add_special_tokens=False),
            },
            "self_trained_wordpiece_cased": {
                "tokenizer": lambda text: self.self_trained_wordpiece_cased.encode(text, add_special_tokens=False),
            },
            "self_trained_wordpiece_uncased": {
                "tokenizer": lambda text: self.self_trained_wordpiece_uncased.encode(text, add_special_tokens=False),
            },
            "word_tokenize": {                
                "tokenizer": word_tokenize,
            },
            "word_tokenize_fasttext": {
                "tokenizer": word_tokenize,
            }, 
        }

class DatasetHelper:
    def __init__(self) -> None:
        self.classification_dataset_info = self.__get_classification_dataset_info()

    def __get_classification_dataset_info(self):
        return {
            "emotion-detection": {
                "file_name": "dataset.pkl",
                "used_columns": ["text"],
                "title_column": None,
                "text_column": "text",
                "label_column": "emotions",
                "title_max_length": {
                    "pre_trained_wordpiece_cased": 0,
                    "pre_trained_wordpiece_uncased": 0,
                    "self_trained_wordpiece_cased": 0,
                    "self_trained_wordpiece_uncased": 0,
                    "word_tokenize": 0,
                },
                "text_max_length": {
                    "pre_trained_wordpiece_cased": 200,
                    "pre_trained_wordpiece_uncased": 200,
                    "self_trained_wordpiece_cased": 200,
                    "self_trained_wordpiece_uncased": 200,
                    "word_tokenize": 200,
                },
            },

            "movie-review": {
                "file_name": "dataset.csv",
                "used_columns": ["text"],
                "title_column": None,
                "text_column": "review",
                "label_column": "sentiment",
                "title_max_length": {
                    "pre_trained_wordpiece_cased": 0,
                    "pre_trained_wordpiece_uncased": 0,
                    "self_trained_wordpiece_cased": 0,
                    "self_trained_wordpiece_uncased": 0,
                    "word_tokenize": 0,
                },
                "text_max_length": {
                    "pre_trained_wordpiece_cased": 1000,
                    "pre_trained_wordpiece_uncased": 1000,
                    "self_trained_wordpiece_cased": 1000,
                    "self_trained_wordpiece_uncased": 1000,
                    "word_tokenize": 1000,
                },
            },

            "news-aggregator": {
                "file_name": "dataset.csv",
                "used_columns": ["title"],
                "title_column": "TITLE",
                "text_column": None,
                "label_column": "CATEGORY",
                "title_max_length": {
                    "pre_trained_wordpiece_cased": 1000,
                    "pre_trained_wordpiece_uncased": 1000,
                    "self_trained_wordpiece_cased": 1000,
                    "self_trained_wordpiece_uncased": 1000,
                    "word_tokenize": 1000, 
                },
                "text_max_length": {
                    "pre_trained_wordpiece_cased": 0,
                    "pre_trained_wordpiece_uncased": 0,
                    "self_trained_wordpiece_cased": 0,
                    "self_trained_wordpiece_uncased": 0,
                    "word_tokenize": 0,
                },
            },

            "sarcasm-detection": {
                "file_name": "dataset.json",
                "used_columns": ["title"],
                "title_column": "headline",
                "text_column": None,
                "label_column": "is_sarcastic",
                "title_max_length": {
                    "pre_trained_wordpiece_cased": 225,
                    "pre_trained_wordpiece_uncased": 225,
                    "self_trained_wordpiece_cased": 225,
                    "self_trained_wordpiece_uncased": 225,
                    "word_tokenize": 225,
                },
                "text_max_length": {
                    "pre_trained_wordpiece_cased": 0,
                    "pre_trained_wordpiece_uncased": 0,
                    "self_trained_wordpiece_cased": 0,
                    "self_trained_wordpiece_uncased": 0,
                    "word_tokenize": 0,
                },
            },
        }