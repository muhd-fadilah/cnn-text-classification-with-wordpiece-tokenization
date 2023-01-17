import os

from src.helpers import PreprocessingHelper
from src.preprocessing import TextCleaning
from tokenizers import BertWordPieceTokenizer

class DataCleaner:
    def __init__(self, source_dir: str) -> None:
        #helper
        preprocessing_helper = PreprocessingHelper()

        #text cleaner
        self.text_cleaner = preprocessing_helper.text_cleaner

        #text cleaning types
        self.text_cleaning_types = preprocessing_helper.text_cleaning_types

        #source files directory
        self.source_dir = source_dir

    def run(self):
        print("Self training data cleaning is in progress...")

        #run preprocessing of wordpiece dataset
        text_cleaning = TextCleaning(
            source_dir=self.source_dir,
            output_dir=self.source_dir,
            text_cleaner=self.text_cleaner,
            text_cleaning_types=self.text_cleaning_types,
            mode="wordpiece",
            title_column="title",
            text_column="content"
        )

        text_cleaning.run()

        print("Nela-2017 dataset preprocessing is done!\n")

class WordpieceTrainer:
    def __init__(
        self, 
        source_file: str,
        clean_text: bool = True,
        vocab_size: int = 30_000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        wordpieces_prefix: str = '##',
        special_tokens: list = ["[UNK]"]
    ):
        #training parameters
        self.source_file = source_file
        self.clean_text = clean_text
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.limit_alphabet = limit_alphabet
        self.wordpieces_prefix = wordpieces_prefix
        self.special_tokens = special_tokens
        
        #path of wordpiece directory
        self.wordpiece_dir = os.path.join(os.getcwd(), "wordpiece")

    def __wordpiece_training(self, output_dir: str, lowercase: bool):
        #create bert wordpiece tokenizer
        tokenizer = BertWordPieceTokenizer(
            clean_text=self.clean_text,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=lowercase
        )

        #train tokenizer
        tokenizer.train(
            files=self.source_file, 
            vocab_size=self.vocab_size, 
            min_frequency=self.min_frequency, 
            limit_alphabet=self.limit_alphabet, 
            wordpieces_prefix=self.wordpieces_prefix, 
            special_tokens=self.special_tokens
        )

        #save tokenizer
        tokenizer.save_model(output_dir)
    
    def run(self):
        for lowercase in [False, True]:
            #type of case
            case_type = "uncased" if lowercase else "cased"

            print(f"{case_type.capitalize()} wordpiece training is in progress...")

            #output directory
            output_dir = os.path.join(self.wordpiece_dir, case_type)

            #create output directory if not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            #train wordpiece tokenizer
            self.__wordpiece_training(output_dir=output_dir, lowercase=lowercase)
        
        print("Wordpiece training is done!\n")

if __name__ == '__main__':
    #directory of nela-2017 dataset
    nela_dir = os.path.join(os.getcwd(), "dataset", "nela-2017")
    source_file = os.path.join(nela_dir, "preprocessed_text.txt")

    #run data cleaner
    data_cleaner = DataCleaner(source_dir=nela_dir)
    data_cleaner.run()

    #run wordpiece trainer
    wordpiece_trainer = WordpieceTrainer(
        source_file=source_file
    )

    wordpiece_trainer.run()