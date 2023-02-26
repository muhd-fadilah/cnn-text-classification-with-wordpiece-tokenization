import pandas as pd
import pickle
import os

from src.helpers import DatasetHelper, PreprocessingHelper
from wordcloud import WordCloud
from nltk.corpus import stopwords

class InspectPreprocessedDataset:
    def __init__(self) -> None:
        self.dataset_path = os.path.join(os.getcwd(), "dataset")
        self.word_cloud_path = os.path.join(os.getcwd(), "word-cloud")
        self.dataset_helper = DatasetHelper()
        self.preprocessing_helper = PreprocessingHelper()
        self.stop_words = set(stopwords.words('english'))

    def count_total_labels(self, target_path):
        with open(os.path.join(target_path, "labels.pkl"), "rb") as f:
            labels = pickle.load(f)    
        
        return len(labels)

    def count_total_data(self, df):
        return len(df.index)

    def create_word_cloud(self, text, result_path):
        text = ' '.join(text).lower()

        word_cloud = WordCloud(stopwords=self.stop_words, min_word_length = 3, collocations=True)
        word_cloud.generate(text)
        word_cloud.to_file(result_path)
    
    def run(self):
        result = list()
        
        for dataset_name in self.dataset_helper.classification_dataset_info.keys():
            current_dataset_path = os.path.join(self.dataset_path, dataset_name)
            df = pd.read_csv(os.path.join(current_dataset_path, "preprocessed_text.csv"))
            total_data = self.count_total_data(df)
            total_labels = self.count_total_labels(current_dataset_path)
            
            temp = dict()
            temp["dataset"] = dataset_name
            temp["total_data"] = total_data
            temp["total_labels"] = total_labels

            result.append(temp)

            for used_column in self.dataset_helper.classification_dataset_info[dataset_name]["used_columns"]:
                for text_cleaning_type in self.preprocessing_helper.text_cleaning_types:
                    current_word_cloud_path = os.path.join(self.word_cloud_path, dataset_name)

                    if not os.path.exists(current_word_cloud_path):
                        os.makedirs(current_word_cloud_path)
                    
                    self.create_word_cloud(
                        df[f"{used_column}_{text_cleaning_type}"].astype(str).tolist(), 
                        os.path.join(current_word_cloud_path, text_cleaning_type + ".png")
                    )

        pd.DataFrame(result).to_csv("preprocessing_result_summary.csv")

if __name__ == '__main__':
    inspect_preprocessed_dataset = InspectPreprocessedDataset()
    inspect_preprocessed_dataset.run()