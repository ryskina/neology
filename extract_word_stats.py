import os
import glob
import math
import random
from collections import Counter, defaultdict
from scipy import stats

from utils import *


class WordStatsExtractor:
    def __init__(self, coha_path, coca_path):
        self.data_path_historical = coha_path   # path to COHA text directory (containing decade subdirs)
        self.data_path_modern = coca_path       # path to COCA text directory (containing genre subdirs)

        self.frequency_dict_historical = {}     # {"total" : {word: frequency}, "1810s" : {word: frequency}, ...}
        self.frequency_dict_modern = {}         # {"total" : {word: frequency}}
        self.capitalization_counter_dict = defaultdict(lambda: Counter())  # {word : {form: count}}

    def extract_frequencies(self, vocabulary, data_split):
        """
        Collecting word frequencies for the specified split ('historical' or 'modern')
        :param vocabulary: vocabulary of nouns for analysis
        :param data_split: 'historical' or 'modern' (corresponding to COHA and COCA respectively)
        :return
        """

        assert data_split == "historical" or data_split == "modern"

        num_tokens_total = 0
        counts_total = Counter()

        if data_split == "historical":
            data_dir = self.data_path_historical
            # Historical data is COHA corpus up to 1989
            subdirs = ["1810s", "1820s", "1830s", "1840s", "1850s",
                       "1860s", "1870s", "1880s", "1890s", "1900s",
                       "1910s", "1920s", "1930s", "1940s", "1950s",
                       "1960s", "1970s", "1980s"]
        else:
            data_dir = self.data_path_modern
            # Modern data is entire COCA corpus
            subdirs = ["text_academic_rpe", "text_fiction_awq",
                       "text_magazine_qch", "text_newspaper_lsp",
                       "text_spoken_kde"]

        for subdir in subdirs:
            current_dir = f"{data_dir}/{subdir}/"
            if not os.path.exists(current_dir):
                print(f"Missing subdirectory: {subdir}")
                continue

            print(f"Processing {current_dir}")
            num_tokens_subdir = 0
            counts_in_subdir = Counter()

            for filename in glob.glob(current_dir + "*.txt"):
                with open(filename, 'r') as fin:
                    for line in fin:
                        tokens = line.strip().split()
                        num_tokens_subdir += len(tokens)
                        for token in tokens:
                            word = token.lower()
                            if word in vocabulary:
                                counts_in_subdir[word] += 1
                                # For each word, we also count its occurrences in different capitalization forms
                                if data_split == "modern":
                                    self.capitalization_counter_dict[word][token] += 1

            if data_split == "historical":
                self.frequency_dict_historical[subdir] = Utils.normalize(counts_in_subdir, num_tokens_subdir)
            num_tokens_total += num_tokens_subdir
            counts_total += counts_in_subdir

            if data_split == "historical":
                self.frequency_dict_historical["total"] = Utils.normalize(counts_total, num_tokens_total)
            else:
                self.frequency_dict_modern["total"] = Utils.normalize(counts_total, num_tokens_total)

    def extract_neologisms(self, outfile):
        """
        Extracting a list of neologisms, filtering by modern-historical frequency ratio, word length and capitalization
        :param outfile: file path path to output neologisms
        :return: list of extracted neologisms
        """
        vocabulary = self.frequency_dict_modern["total"].keys()
        neologism_counter = Counter()

        for word in vocabulary:
            # To be included as a neologism, the word has to occur most frequently in lowercase
            most_common_form = self.capitalization_counter_dict[word].most_common(1)[0][0]
            if most_common_form != word:
                continue
            freq_mod = self.frequency_dict_modern["total"][word]
            freq_hist = self.frequency_dict_historical["total"][word]

            # First, we select words that are MIN_FREQUENCY_RATIO times more frequent in modern than historical data
            if freq_mod > 0 and (freq_hist == 0 or freq_mod / freq_hist > MIN_FREQUENCY_RATIO):
                neologism_counter[word] = freq_mod

        # We filter out words shorter than MIN_WORD_LEN characters, and only leave the first 1000
        neologism_list = [n[0] for n in neologism_counter.most_common() if len(n[0]) >= MIN_WORD_LEN][:1000]
        with open(outfile, 'w') as fout:
            for word in neologism_list:
                fout.write(f"{word}\n")
                fout.flush()

        return neologism_list

    def extract_frequency_growth(self, vocabulary, outfile):
        """
        For all words in the vocabulary, compute their frequency growth rate (Spearman correlation
        between the decade time series and frequency time series by decade in the historical data)
        :param vocabulary: vocabulary of nouns for analysis
        :param outfile: file path to output words and their frequency growth rates
        :return: word - frequency growth rate dictionary
        """

        # Time steps enumerate decades from 1810s to 1880s
        time_steps = list(range(1, 19))
        fout = open(outfile, 'w')
        frequency_growth_dict = {}

        for word in vocabulary:
            word_frequency_series = []
            for key in sorted(self.frequency_dict_historical.keys())[:18]:
                frequency = self.frequency_dict_historical[key].get(word, 0)
                word_frequency_series.append(frequency)
            if sum(word_frequency_series) > 0:
                corr, pval = stats.spearmanr(time_steps, word_frequency_series)
                fout.write(f"{word}\t{corr}\t{pval}\n")
                frequency_growth_dict[word] = corr

        fout.close()
        return frequency_growth_dict

    def pair_neologisms_with_controls(self, frequency_growth_dict, neologism_list, outfile,
                                      stability_constraint=True, seed=None):
        """
        Collecting a set of control words by pairing each neologism with a control counterpart,
        controlling for overall frequency and word length
        :param frequency_growth_dict: word - frequency growth rate dictionary
        :param neologism_list: list of neologisms to pair
        :param outfile: file path to output neologism - control word pairs
        :param stability_constraint: toggles between 'stable' and 'relaxed' control sets (default = True)
        :param seed: seed to randomize the relazed control sets (default = None)
        :return: neologism - control word pair dictionary
        """
        pairs_dict = {}

        candidate_controls = []
        frequencies_historical = self.frequency_dict_historical['total']
        frequencies_modern = self.frequency_dict_modern['total']

        for word, frequency_growth in frequency_growth_dict.items():
            if word not in neologism_list:
                if not stability_constraint:
                    candidate_controls.append(word)
                if stability_constraint and math.fabs(float(frequency_growth)) < MAX_SPEARMANS_CORRELATION:
                    candidate_controls.append(word)

        print(f"Found {len(candidate_controls)} candidate control words")

        if seed is not None:
            random.seed(seed)
            random.shuffle(candidate_controls)

        for neologism in neologism_list:
            neologism_len = len(neologism)
            neologism_freq = frequencies_modern.get(neologism, 0)

            matched_flag = False
            for control in candidate_controls:
                control_len = len(control)
                control_freq = frequencies_historical.get(control, 0)
                if control_freq == 0:
                    continue
                freq_ratio = neologism_freq / control_freq

                # To replicate our results, one needs to restrict the frequency ratio to fall within (0.75, 1.33),
                # rather than (0.75, 1.25) as reported in the paper

                if abs(neologism_len - control_len) < 2 and 0.75 < freq_ratio < 1.33:
                    pairs_dict[neologism] = control
                    candidate_controls.remove(control)
                    matched_flag = True
                    break

            if not matched_flag:
                print(f"Failed to pair with a control: {neologism}")

        print(f"Created {len(pairs_dict)} neologism-control pairs")
        with open(outfile, 'w') as fout:
            for neologism in pairs_dict.keys():
                 fout.write("\t".join([neologism, pairs_dict[neologism]]) + "\n")

        return pairs_dict
