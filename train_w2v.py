from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
import os
import argparse
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str,
                        help="Path to COHA or COCA text directory (containing decade / genre subdirectories)")
    parser.add_argument("data_split", choices=["historical", "modern"],
                        help="Historical (COHA) or Modern (COCA) data split")
    return parser.parse_args()


class EmbeddingTrainer:
    def __init__(self, params):
        self.data_path = params.data_path
        self.model_dirname = params.model_dirname
        self.data_split = params.data_split
        self.model_file_path = f"models/{self.data_split}.w2v.bin"
        self.sentences = []

    def load_sentences(self):
        """
        Reading text files and converting them to a set of sentence that the embeddings will be trained on
        :return:
        """
        if self.data_split == 'historical':
            dirs = ['1810s', '1820s', '1830s', '1840s', '1850s',
                    '1860s', '1870s', '1880s', '1890s', '1900s',
                    '1910s', '1920s', '1930s', '1940s', '1950s',
                    '1960s', '1970s', '1980s']
        else:
            dirs = ['text_academic_rpe', 'text_fiction_awq', 'text_magazine_qch', 'text_newspaper_lsp',
                    'text_spoken_kde']
        for dirname in dirs:
            print(f"Reading directory: {dirname}", flush=True)
            files = os.listdir(f"{self.data_path}/{dirname}")
            for filename in files:
                with open(f"{self.data_path}/{dirname}/{filename}") as f:
                    if self.data_split == 'historical':
                        # specific to reading COHA files
                        lines = f.readlines()
                        if len(lines) != 3:
                            print(f"File {filename} contains {len(lines)} lines", flush=True)
                            continue
                        texts = [lines[2]]
                    else:
                        # specific to reading COCA files
                        texts = f.readlines()[1:]
                    for text in texts:
                        try:
                            sents = sent_tokenize(text)
                        except UnicodeDecodeError:
                            print(f"UnicodeDecodeError occurred in {filename}")
                            sents = []
                        for sent in sents:
                            self.sentences.append([x.lower() for x in sent.split(' ')
                                                   if x != '@' and x.lower() != '<p>'])

    def train_w2v(self):
        """
        Learning Word2Vec embeddings from the provided data
        :return:
        """
        self.load_sentences()
        print(f"Building Word2Vec embeddings for {self.data_split.upper()} data", flush=True)
        print(f"Model will be saved to file {self.model_file_path}", flush=True)

        model = Word2Vec(self.sentences, size=300, window=5, min_count=5, workers=multiprocessing.cpu_count(), sg=1)
        print(f"Finished training Word2Vec for {self.data_split.upper()} data", flush=True)
        model.save(self.model_file_path)
        print(f"Model saved to file", flush=True)


# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    trainer = EmbeddingTrainer(args)
    trainer.train_w2v()
