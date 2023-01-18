import pandas as pd
import seaborn as sns
import os

from src.helpers import DatasetHelper, PreprocessingHelper

class InspectLength:
    def __init__(self) -> None:
        #helpers
        dataset_helper = DatasetHelper()
        preprocessing_helper = PreprocessingHelper()
        
        #classification dataset info
        self.classification_dataset_info = dataset_helper.classification_dataset_info

        #text cleaner
        self.text_cleaner = preprocessing_helper.text_cleaner

        #tokenizer types
        self.tokenizer_types = preprocessing_helper.get_tokenizer_types()

    def __token_distplot(
        self,
        dataframe: pd.DataFrame, 
        column: str, 
        tokenizer, 
        output_dir: str,
        max_length: int,
    ):
        token_lengths = []

        #iterate over text
        for text in dataframe[column]:
            #get tokens
            tokens = tokenizer(text)

            #add current token length to list
            token_lengths.append(min(len(tokens), max_length))

        #draw distplot of token length
        sns_plot = sns.displot(token_lengths, kind='kde')
        
        #save distplot to output directory
        sns_plot.figure.savefig(output_dir)
    
    def run(self):
        #directory of length distplot results
        length_distplots_dir = os.path.join(os.getcwd(), "length-distplots")
        
        #create length distplots directory if not exists
        if not os.path.exists(length_distplots_dir):
            os.makedirs(length_distplots_dir)

        print("Getting the length distplots...")

        #iterate over dataset name
        for dataset_name in self.classification_dataset_info:
            #iterate over used columns
            for column in self.classification_dataset_info[dataset_name]["used_columns"]:
                for tokenizer_type in self.tokenizer_types.keys():
                    #skip if using pretrained fasttext
                    if "_fasttext" in tokenizer_type:
                        continue
                    
                    #get file name
                    file_name = self.classification_dataset_info[dataset_name]["file_name"]
                    
                    #get dataset file
                    dataset_dir = os.path.join(os.getcwd(), "dataset", dataset_name, file_name)
                    
                    #define initial value of dataframe
                    dataframe = None

                    #load dataframe if given file has csv extension
                    if(file_name.endswith(".csv")):
                        dataframe = pd.read_csv(dataset_dir)

                    #load dataframe if given file has json extension
                    elif(file_name.endswith(".json")):
                        dataframe = pd.read_json(dataset_dir, lines=True)

                    #load dataframe if given file has csv extension
                    elif(file_name.endswith(".pkl")):
                        dataframe = pd.read_pickle(dataset_dir)

                    #raise exception if source file format is not supported
                    else:
                        raise Exception("Unsupported source file format! The supported formats: '*.csv,' '*.json'")

                    #directory of distplot output
                    distplot_dir = os.path.join(length_distplots_dir, dataset_name)
                    
                    #create directory of distplot output if not exist
                    if not os.path.exists(distplot_dir):
                        os.makedirs(distplot_dir)

                    #draw token distplot
                    self.__token_distplot(
                        dataframe=dataframe, 
                        column=self.classification_dataset_info[dataset_name][f"{column}_column"], 
                        tokenizer=self.tokenizer_types[tokenizer_type]["tokenizer"], 
                        output_dir=os.path.join(distplot_dir, f"{tokenizer_type}_{column}.png"),
                        max_length=1000
                    )
        
        print("Done!")

if __name__ == '__main__':
    #run inspect length
    inspect_length = InspectLength()
    inspect_length.run()