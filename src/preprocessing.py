import json
import pandas as pd
import os
import pickle
import numpy as np
import nltk
import math
import gc

from typing import Literal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class TextCleaning:
    def __init__(
        self, 
        source_dir: str, 
        output_dir: str,
        text_cleaner,
        text_cleaning_types,
        mode: Literal["self_training", "classification"], 
        title_column: str | None = None,
        text_column: str | None = None, 
        label_column: str | None = None
    ):
        #text cleaner
        self.text_cleaner = text_cleaner

        #text cleaning types
        self.text_cleaning_types = text_cleaning_types
        
        #preprocessing mode
        self.mode = mode

        #source directory or file
        self.source_dir = source_dir

        #target directory for preprocessing result
        self.output_dir = output_dir

        #title column name
        self.title_column = title_column

        #text column name
        self.text_column = text_column

        #label column name
        self.label_column = label_column

    def __load_classification_dataframe(self):
        #load dataframe if given file has csv extension
        if self.source_dir.endswith(".csv"):
            self.df = pd.read_csv(self.source_dir)

        #load dataframe if given file has json extension
        elif self.source_dir.endswith(".json"):
            self.df = pd.read_json(self.source_dir, lines=True)

        #load dataframe if given file has pkl extension
        elif self.source_dir.endswith(".pkl"):
            self.df = pd.read_pickle(self.source_dir)

        #raise exception if source file format is not supported
        else:
            raise Exception("Unsupported source file format! The supported formats: '*.csv,' '*.json'")

    def __load_self_training_data(self):
        #empty dictionary for result
        data = []

        #iterate over directories
        for month in os.scandir(self.source_dir):
            #ignore python files
            if not month.is_dir():
                continue

            for date in os.listdir(os.path.join(self.source_dir, month.path)):
                for source in os.listdir(os.path.join(self.source_dir, month, date)):
                    for article in os.listdir(os.path.join(self.source_dir, month, date, source)):
                        #open article file
                        with open(os.path.join(self.source_dir, month, date, source, article)) as f:
                            #read article file
                            json_txt = f.read()
                            
                            #handling exception for json loading
                            try:
                                article_dict = json.loads(json_txt)
                            except json.decoder.JSONDecodeError:
                                continue
                            
                            #create dictionary for current article
                            article_dict = {
                                "whole": article_dict[self.title_column] + '. ' + article_dict[self.text_column],
                            }

                            #add current article dictionary to result
                            data.append(article_dict)
        
        #create dataframe from result
        self.df = pd.DataFrame(data)
    
    def __export_self_training_dataset(self):
        self.__load_self_training_data()

        #write preprocessing result on txt file
        with open(os.path.join(self.output_dir, "preprocessed_text.txt"), 'w') as f:
            self.df["whole"].apply(self.text_cleaner.txt_file, txt_instance=f)

    def __export_classification_dataset(self):
        #load source dataframe
        self.__load_classification_dataframe()
        
        #dataframe for processing result
        preprocessing_result = pd.DataFrame()

        #apply preprocessing to title column
        if self.title_column is not None:
            preprocessing_result = self.text_cleaner.classification_dataset(self.df, preprocessing_result, self.title_column, "title")

        #apply preprocessing to text column
        if self.text_column is not None:
            preprocessing_result = self.text_cleaner.classification_dataset(self.df, preprocessing_result, self.text_column, "text")

        #merge title and text
        if self.title_column and self.text_column:
            for text_cleaning_type in self.text_cleaning_types.keys():
                preprocessing_result[f"whole_{text_cleaning_type}"] = preprocessing_result[f"title_{text_cleaning_type}"].astype(str) + preprocessing_result[f"text_{text_cleaning_type}"].astype(str)
        
        #insert data label
        preprocessing_result["label"] = self.df[self.label_column]
        
        #convert string label to numeric
        label_encoder = preprocessing.LabelEncoder()
        preprocessing_result["label"] = label_encoder.fit_transform(preprocessing_result["label"])
        labels = list(label_encoder.classes_)
    
        #save labels info
        with open(os.path.join(self.output_dir, "labels.pkl"), "wb") as f:
            pickle.dump(labels, f)

        #save preprocessed dataset to csv
        preprocessing_result.to_csv(os.path.join(self.output_dir, "preprocessed_text.csv"), index=False, encoding="utf-8")

    def run(self):
        #export as classification dataset if mode is 'classification'
        if self.mode == 'classification':
            if not self.label_column:
                raise Exception("Please define label column name!")

            elif not (self.title_column or self.text_column):
                raise Exception("Please define at least one column name of string type to be used!")

            self.__export_classification_dataset()
            return

        if not (self.title_column and self.text_column):
            raise Exception("Please define title and text columns name!")

        #export as self training dataset if mode is 'self_training'
        self.__export_self_training_dataset()

class Tokenization:
    def __init__(
        self, 
        source_dir: str, 
        output_dir: str,
        text_cleaning_types,
        tokenizer_types,
        padding_index: int,
        random_state: int,
        dataset_info: dict,
        split_size: float
    ):
        #text_cleaning types
        self.text_cleaning_types = text_cleaning_types
        
        #tokenizer types
        self.tokenizer_types = tokenizer_types

        #text cleaning result dataframe
        self.df = pd.read_csv(source_dir, index_col=False)

        #output directory
        self.output_dir = output_dir
        
        #padding index
        self.padding_index = padding_index
        
        #random state
        self.random_state = random_state

        #dictionary contains dataset info
        self.dataset_info = dataset_info

        #vocabulary dictionary
        self.vocabulary = dict()

        #dataset split size
        self.split_size = split_size

    def __add_padding(self, text: list, max_length: int):
        #return nan if text is empty
        if text == list():
            return np.nan
        
        #truncate if index conversion result is longer than max_length
        if(len(text) > max_length):
            return text[:max_length]

        #add padding until position of max_length
        while len(text) < max_length:
            text.insert(len(text), self.padding_index)
        
        return text

    def __word_to_index(self, text: list, vocabulary: dict):
        #convert word to index
        return [vocabulary.get(word, vocabulary["[UNK]"]) for word in text]

    def __tokenize_dataframe(
        self,
        tokenizer_type: str,
        text_cleaning_type: str,
        vocabulary: dict
    ):
        #result dataframe
        result = pd.DataFrame()

        #tokenizer
        tokenizer = self.tokenizer_types[tokenizer_type]["tokenizer"]

        #used columns
        used_columns = self.dataset_info["used_columns"]

        #tokenize text columns
        for column in used_columns:
            #tokenize text in column
            result[column] = self.df[column + "_" + text_cleaning_type].apply(tokenizer)

            #convert tokenized text to index
            result[column] = result[column].apply(
                self.__word_to_index, 
                vocabulary=vocabulary,
            )

            #max length
            dataset_max_length = self.dataset_info[f"{column}_max_length"][tokenizer_type]
            current_result_max_length = result[column].str.len().max()
            max_length = min(dataset_max_length, current_result_max_length)

            #add padding
            result[column] = result[column].apply(self.__add_padding, max_length=max_length)

        return result

    def __split_dataset(self, result: pd.DataFrame, labels: list):
        test_size = math.floor(len(result) * self.split_size)
        
        #split training and testing dataset
        x_train, x_val, y_train, y_val = train_test_split(result, labels, test_size=test_size, random_state=self.random_state)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=self.random_state)
    
        return x_train, x_val, x_test, y_train, y_val, y_test

    def __save_as_pickle(self, data: dict, path: str):
        for file_name in data.keys():
            #save as pickle
            with open(os.path.join(path, f"{file_name}.pkl"), "wb") as f:
                pickle.dump(data[file_name], f)

    def __build_vocabulary(self, vocab_column: str, tokenizer_type: str, text_cleaning_type: str):
        #used column
        column = f"{vocab_column}_{text_cleaning_type}"
        
        #tokenizer
        tokenizer = self.tokenizer_types[tokenizer_type]["tokenizer"]
        
        #current max length
        current_max_length =  self.df[column].str.len().max()

        #create list of tokenized text
        list_of_tokenized_text = self.df[column].apply(tokenizer).tolist()

        #dataset max length
        dataset_max_length = self.dataset_info["title_max_length"][tokenizer_type] + self.dataset_info["text_max_length"][tokenizer_type]

        #used max length
        max_length = min(dataset_max_length, current_max_length)

        #create frequency distribution object
        vocabulary = dict()
        
        #add padding index to vocabulary
        vocabulary['[PAD]'] = self.padding_index

        #add unknown index to vocabulary
        vocabulary['[UNK]'] = self.padding_index + 1

        #starting index after reserved indices
        index = self.padding_index + 2

        #iterate over list of text
        for text in list_of_tokenized_text:
            #truncate if text longer than max length
            if len(text) > max_length:
                text = text[:max_length]
            
            #iterate over words
            for word in text:
                if word not in vocabulary:
                    #add word index if not exist in vocabulary
                    vocabulary[word] = index
                    index += 1
                
        return vocabulary

    def run(self):
        #enable garbage collector
        gc.enable()

        #dropping any nan value
        self.df.dropna(inplace=True)

        #column uses to build vocabulary
        vocab_column = "whole"

        if self.dataset_info["title_column"] is None:
            vocab_column = "text"

        elif self.dataset_info["text_column"] is None:
            vocab_column = "title"

        for text_cleaning_type in self.text_cleaning_types.keys():
            for tokenizer_type in self.text_cleaning_types[text_cleaning_type]["tokenizer_types"]:
                #skip if using pre-trained fasttext
                if 'fasttext' in tokenizer_type:
                    continue

                #build vocabularies
                vocab = self.__build_vocabulary(
                    vocab_column=vocab_column,
                    tokenizer_type=tokenizer_type,
                    text_cleaning_type=text_cleaning_type
                )

                #tokenize dataframe
                text = self.__tokenize_dataframe(
                    tokenizer_type=tokenizer_type,
                    text_cleaning_type=text_cleaning_type,
                    vocabulary=vocab
                )

                #dropping any nan value, including its label
                text["label"] = self.df["label"]
                text.dropna(inplace=True)

                #get remaining labels
                labels = text["label"].tolist()

                #drop label column from text dataframe
                text.drop(["label"], axis=1, inplace=True)

                #split dataset
                x_train, x_val, x_test, y_train, y_val, y_test = self.__split_dataset(text, labels)

                data = {
                    "x_train": x_train.to_dict('list'),
                    "x_val": x_val.to_dict('list'),
                    "x_test": x_test.to_dict('list'),
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                    "vocabulary": vocab
                }

                #output directory
                output_dir = os.path.join(self.output_dir, f"{tokenizer_type}_{text_cleaning_type}")

                #create directory if not exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                #save dataset and vocabulary
                self.__save_as_pickle(data, output_dir)

                #remove unused objects
                del data
                del text
                del labels
                del vocab

                #collect garbage
                gc.collect()

        #remove unused objects
        del self.df

        #collect garbage
        gc.collect()