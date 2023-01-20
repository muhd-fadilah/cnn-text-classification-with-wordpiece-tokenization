import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
import gc

from src.preprocessing import TextCleaning, Tokenization
from src.helpers import PreprocessingHelper, DatasetHelper
from tqdm import tqdm
from src.model import TextClassifier
from src.run import Run

class Controller:
    def __init__(self):
        #helpers
        dataset_helper = DatasetHelper()
        preprocessing_helper = PreprocessingHelper()
        
        #classification dataset info
        self.classification_dataset_info = dataset_helper.classification_dataset_info

        #text cleaner
        self.text_cleaner = preprocessing_helper.text_cleaner

        #text cleaning types
        self.text_cleaning_types = preprocessing_helper.text_cleaning_types

        #tokenizer types
        self.tokenizer_types = preprocessing_helper.get_tokenizer_types()

        #path of dataset directory
        self.dataset_dir = os.path.join(os.getcwd(), "dataset")
        
        #random state
        self.random_state = 719

        #parameters related to model
        self.embedding_dim = 300
        self.padding_index = 0
        self.filter_sizes = [3, 4, 5]
        self.num_filters = [100, 100, 100]
        self.dropout = 0.3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #parameter related to model training
        self.learning_rate = 0.01
        self.epochs = 10
        self.batch_size = 30

    def __set_random_seed(self):
        #set random seed on various libraries
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)

    def __classification_dataset_preprocessing(self):
        #preprocessing of classification dataset
        for dataset_name in tqdm(self.classification_dataset_info.keys(), desc="Classification dataset preprocessing"):
            #directory of current dataset
            source_dataset_dir = os.path.join(self.dataset_dir, dataset_name)

            #current dataset information
            current_dataset_info = self.classification_dataset_info[dataset_name]
            
            #run text cleaning on current dataset
            text_cleaning = TextCleaning(
                source_dir=os.path.join(source_dataset_dir, self.classification_dataset_info[dataset_name]["file_name"]),
                output_dir=os.path.join(source_dataset_dir),
                text_cleaner=self.text_cleaner,
                text_cleaning_types=self.text_cleaning_types,
                mode="classification",
                title_column=current_dataset_info["title_column"],
                text_column=current_dataset_info["text_column"],
                label_column=current_dataset_info["label_column"]
            )

            text_cleaning.run()
            
            #run tokenization on current dataset
            tokenization = Tokenization(
                source_dir=os.path.join(source_dataset_dir, "preprocessed_text.csv"),
                output_dir=source_dataset_dir,
                text_cleaning_types=self.text_cleaning_types,
                tokenizer_types=self.tokenizer_types,
                padding_index=self.padding_index,
                random_state=self.random_state,
                dataset_info=current_dataset_info,
            )

            tokenization.run()

    def __get_labels(self, dataset_name):
        #open pickle file
        with open(os.path.join(self.dataset_dir, dataset_name, "labels.pkl"), "rb") as f:
            labels = pickle.load(f)

        return labels

    def __get_vocab(self, dataset_name, preprocessing_type):
        #open pickle file
        with open(os.path.join(self.dataset_dir, dataset_name, preprocessing_type, "vocabulary.pkl"), "rb") as f:
            vocab = pickle.load(f)

        return vocab

    def __get_pre_trained_fasttext_embedding(self, vocabulary):
        #open fasttext vec file
        with open(os.path.join(os.getcwd(), "fasttext", "pre_trained_fasttext.vec"), 'r', encoding='utf-8', newline="\n", errors='ignore') as f:
            #get embedding size
            _, d = map(int, f.readline().split())
            
            #initialize random embeddings
            embeddings = np.random.uniform(-0.25, 0.25, (len(vocabulary), d))
            embeddings[vocabulary['[PAD]']] = np.zeros((d,))

            #load pre-trained vectors
            for line in f:
                #get tokens
                tokens = line.rstrip().split(' ')
                
                #index '0' of tokens is word
                word = tokens[0]

                #if current word exist in vocabulary, use its embedding value
                if word in vocabulary:
                    embeddings[vocabulary[word]] = np.array(tokens[1:], dtype=np.float32)

            return torch.tensor(embeddings)

    def __get_total_experiments(self):
        total = 0

        #iterate over combination possibilities
        for dataset_name in self.classification_dataset_info.keys():
            for _ in self.classification_dataset_info[dataset_name]["used_columns"]:
                for text_cleaning_type in self.text_cleaning_types.keys():
                    for _ in self.text_cleaning_types[text_cleaning_type]["tokenizer_types"]:
                        total += 1

        return total
        
    def __train_models(self):
        #enable garbage collector
        gc.enable()

        #total experiments
        total_experiments = self.__get_total_experiments()

        #summary output path
        summary_result_path = os.path.join(os.getcwd(), "result_summary.csv")

        with tqdm(total=total_experiments, desc="Experiment progress") as pbar:
            #iterate over combination possibilities
            for dataset_name in self.classification_dataset_info.keys():
                #get labels for current dataset
                labels = self.__get_labels(dataset_name)
                
                #load preprocessed dataset
                preprocessed_data = self.__load_preprocessed_dataset(dataset_name)

                for text_cleaning_type in self.text_cleaning_types.keys():
                    for tokenizer_type in self.text_cleaning_types[text_cleaning_type]["tokenizer_types"]:
                        #build experiment case name
                        experiment_case = f"{tokenizer_type.removesuffix('_fasttext')}_{text_cleaning_type}"

                        #check if pre-trained FastText is being used
                        use_pre_trained_fasttext = f"{tokenizer_type}_{text_cleaning_type}" != experiment_case

                        #splitted dataset for training, validation, and testing
                        split_types = preprocessed_data[experiment_case]
                        
                        #iterate over used columns
                        for used_column in self.classification_dataset_info[dataset_name]["used_columns"]:
                            data = dict()

                            #build training and testing data
                            data["x_train"] = split_types["x_train"][used_column]
                            data["x_val"] = split_types["x_val"][used_column]
                            data["x_test"] = split_types["x_test"][used_column]
                            data["y_train"] = split_types["y_train"]
                            data["y_val"] = split_types["y_val"]
                            data["y_test"] = split_types["y_test"]

                            #remove split types
                            del split_types

                            #get vocabulary
                            vocab = self.__get_vocab(dataset_name, experiment_case)                
                            
                            #create model
                            model = TextClassifier(
                                num_classes=len(labels), 
                                embedding_dim=self.embedding_dim if not use_pre_trained_fasttext else None,
                                padding_index=self.padding_index,
                                pretrained_embedding=None if not use_pre_trained_fasttext else self.__get_pre_trained_fasttext_embedding(vocab),
                                freeze_embedding=use_pre_trained_fasttext,
                                vocab_size=len(vocab.keys()) if not use_pre_trained_fasttext else None,
                                filter_sizes=self.filter_sizes,
                                num_filters=self.num_filters,
                                dropout=self.dropout
                            )

                            #move model to device
                            model = model.to(self.device)

                            #directory for current experiment case
                            experiment_case_dir = os.path.join(os.getcwd(), "dataset", dataset_name, experiment_case)
                            
                            #directory for best model output
                            model_dir = os.path.join(experiment_case_dir, f"best_{used_column}_{'fasttext_'if use_pre_trained_fasttext else ''}model.bin")

                            #train model to get best accuracy
                            best_val_accuracy = Run.train(
                                model=model,
                                device=self.device,
                                data=data,
                                learning_rate=self.learning_rate,
                                save_path=model_dir,
                                epochs=self.epochs,
                                batch_size=self.batch_size
                            )

                            #remove training model
                            del model

                            #create test model
                            test_model = TextClassifier(
                                num_classes=len(labels), 
                                embedding_dim=self.embedding_dim if not use_pre_trained_fasttext else None,
                                padding_index=self.padding_index,
                                pretrained_embedding=None if not use_pre_trained_fasttext else self.__get_pre_trained_fasttext_embedding(vocab),
                                freeze_embedding=use_pre_trained_fasttext,
                                vocab_size=len(vocab.keys()) if not use_pre_trained_fasttext else None,
                                filter_sizes=self.filter_sizes,
                                num_filters=self.num_filters,
                                dropout=self.dropout
                            )

                            #remove vocab
                            del vocab

                            #move best model to device
                            test_model = test_model.to(self.device)

                            #load best model
                            test_model.load_state_dict(torch.load(model_dir))

                            #get real values and prediction out of best model
                            real_values, predictions = Run.get_predictions(
                                model=test_model,
                                data=data,
                                device=self.device 
                            )
                            
                            #remove data
                            del data

                            #remove test model
                            del test_model

                            #create classification report
                            report = Run.create_classification_report(
                                real_values=real_values, 
                                predictions=predictions, 
                                class_names=labels,
                            )  

                            #remove real values
                            del real_values

                            #remove predictions
                            del predictions        

                            #save dataframe to csv file
                            report.to_csv(os.path.join(experiment_case_dir, f"{used_column}_{'fasttext_' if use_pre_trained_fasttext else ''}classification_report.csv"), encoding="utf-8")

                            #current result dictionary
                            current_result = {
                                "dataset_name": dataset_name,
                                "tokenizer_type": f"{tokenizer_type}{'_fasttext' if use_pre_trained_fasttext else ''}",
                                "text_cleaning_type": text_cleaning_type,
                                "used_column": used_column,
                                "val_accuracy": best_val_accuracy,
                                "test_accuracy": report.loc["accuracy", "f1-score"],
                                "weighted_avg_precision": report.loc["weighted avg", "precision"],
                                "weighted_avg_recall": report.loc["weighted avg", "recall"],
                                "weighted_avg_f1_score": report.loc["weighted avg", "f1-score"],
                            }

                            #dataframe of current result
                            summary_result_update = pd.DataFrame([current_result])
                            
                            #save dataframe to csv file
                            summary_result_update.to_csv(
                                summary_result_path, 
                                header=not os.path.exists(summary_result_path), 
                                index=False, 
                                mode='a', 
                                encoding="utf-8"
                            )

                            #remove unused objects
                            del current_result
                            del summary_result_update

                            #update progress bar
                            pbar.update(1)

                            #print best test accuracy of last training
                            pbar.set_postfix({"prev_test_acc": report.loc["accuracy", "f1-score"]})

                            #remove report
                            del report

                            #collect garbage
                            gc.collect()

                #remove unused objects
                del labels
                del preprocessed_data
                    
                #collect garbage
                gc.collect()

    def __load_preprocessed_dataset(self, dataset_name):
        data = dict()

        #scan through dataset directories
        for path in os.scandir(os.path.join(self.dataset_dir, dataset_name)):
            #skip if not a directory or is a bin file
            if not path.is_dir():
                continue
            
            data[path.name] = dict()

            #scan through all file in current directory
            for file in os.scandir(path):
                #skip vocabulary file
                if not file.name.endswith(".pkl") or file.name == "vocabulary.pkl":
                    continue

                #insert dataset to dictionary
                with open(file, "rb") as f:
                    data[path.name][file.name.removesuffix(".pkl")] = pickle.load(f)

        return data

    def run(self):
        #set random seed
        self.__set_random_seed()

        #run preprocessing on wordpiece and classification dataset
        self.__classification_dataset_preprocessing()

        #run training and validation of models
        self.__train_models()

if __name__ == '__main__':
    #run controller
    controller = Controller()
    controller.run()