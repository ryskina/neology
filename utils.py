import numpy as np
import matplotlib.pyplot as plt

# Fixed hyperparameters used in our analysis
MIN_FREQUENCY_RATIO = 20
MIN_WORD_LEN = 3
MAX_SPEARMANS_CORRELATION = 0.1
COSINE_RADIUS_RANGE = np.arange(0.55, 0.35, -0.025)
EUCLIDEAN_RADIUS_RANGE = np.arange(2, 5.5, 0.5)


class Utils:
    @staticmethod
    def read_vocabulary(lexicon_path):
        """
        Reading the vocabulary of nouns extracted from Wikicorpus
        :param lexicon_path: path to the text file containing the vocabulary of nouns
        :return: vocabulry of nouns for analysis
        """

        nn_lexicon = {}
        with open(lexicon_path) as f:
            for line in f:
                word = line.strip()
                if word.replace('-', '').isalpha():
                    nn_lexicon[word] = ''
        return nn_lexicon

    @staticmethod
    def normalize(d, total):
        """
        Converting raw counts to frequencies
        :param d: collection of counts to be normalized
        :param total: normalization constant
        :return: normalized version of the input
        """

        obj_type = type(d)
        dn = obj_type()
        for key in d.keys():
            dn[key] = float(d[key]) / total
        return dn

    @staticmethod
    def reformat_feats_for_glm(density_filename, growth_filename, radius_range, outfile):
        """
        Reading density and frequency growth values from corresponding files and re-formatting them
        to create input for the GLM script
        :param density_filename: path to the file storing the neighborhood densities
        :param growth_filename: path to the file storing the neighborhood frequency growth rates
        :param radius_range: range of cosine similarity thresholds defining neighborhood sizes
        :param outfile: file path to output features that GLM will be fit to
        :return:
        """

        neologisms_list = []
        controls_list = []
        density_dict = {}
        growth_dict = {}

        with open(density_filename) as fin:
            for line in fin:
                s = line.strip().split('\t')
                neologism = s[0]
                neologisms_list.append(neologism)
                density_dict[neologism] = s[1:len(radius_range)+1]
                control = s[len(radius_range)+1]
                controls_list.append(control)
                density_dict[control] = s[len(radius_range)+2:]

        with open(growth_filename) as fin:
            for line in fin:
                s = line.strip().split('\t')
                neologism = s[0]
                neologisms_list.append(neologism)
                growth_dict[neologism] = s[1:len(radius_range)+1]
                control = s[len(radius_range)+1]
                controls_list.append(control)
                growth_dict[control] = s[len(radius_range)+2:]

        with open(outfile, 'w') as fout:
            vars = ["Word"] + ["DensityAtRadius" + "{0:.3f}".format(r) for r in radius_range] + \
                   ["SpearmanAtRadius" + "{0:.3f}".format(r) for r in radius_range] + ["IsNeologism"]
            fout.write(",".join(vars) + "\n")

            for neologism in neologisms_list:
                feats = [neologism] + [str(x) for x in density_dict[neologism]] + \
                       [str(x) for x in growth_dict[neologism]] + ['1']
                fout.write(",".join(feats) + '\n')
                fout.flush()
            for control in controls_list:
                feats = [control] + [str(x) for x in density_dict[control]] + \
                        [str(x) for x in growth_dict[control]] + ['0']
                fout.write(",".join(feats) + "\n")
                fout.flush()

    @staticmethod
    def plot_neighborhood_stats(mean_neologism_statistic, mean_control_statistic, distance, statistic):
        """
        Visualizing a bar chart of mean neighborhood density or frequency growth rate
        :param mean_neologism_statistic: a list of values of the statistic for different neighborhood sizes,
        averaged over all neologisms
        :param mean_control_statistic: a list of values of the statistic for different neighborhood sizes,
        averaged over all control words
        :param distance: distance metric to use ('cosine' or 'euclidean')
        :param statistic: type of the statistic provided ('density' or 'growth')
        :return:
        """

        assert distance == "cosine" or distance == "euclidean"
        assert statistic == "density" or statistic == "growth"

        if distance == "cosine":
            width = 0.025 / 3
            radius_range = COSINE_RADIUS_RANGE
        else:
            width = 0.5 / 3
            radius_range = EUCLIDEAN_RADIUS_RANGE

        fig, ax = plt.subplots()
        p_neologism = ax.bar(radius_range, mean_neologism_statistic, width, color='b')
        p_control = ax.bar(radius_range + width, mean_control_statistic, width, color='r')
        ax.legend((p_neologism[0], p_control[0]), ("Neigborhoods of neologisms", "Neighborhoods of control words"))
        if statistic == "density":
            ax.set_title("Average number of neighbor words in radius")
        else:
            ax.set_title("Average frequency growth rate of the neighbor words")
        plt.show()
