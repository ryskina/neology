from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist

from utils import *
from projection import *


class NeighborhoodStatsExtractor:
    def __init__(self, historical_model_file_path, modern_model_file_path, vocabulary, spearmanr_dict):
        """
        Loading and aligning the embedding models
        :param historical_model_file_path: path to the historical (COHA) embedding model Word2Vec .bin file
        :param modern_model_file_path: path to the modern (COCA) embedding model Word2Vec .bin file
        :param vocabulary: vocabulary of nouns for analysis
        :param spearmanr_dict: word - frequency growth rate dictionary
        """
        self._word_pairs = {}
        self.vocabulary = vocabulary
        self.spearmanr_dict = spearmanr_dict

        self.model_historical = Word2Vec.load(historical_model_file_path)
        self.model_modern = Word2Vec.load(modern_model_file_path)
        self.model_modern_projected = smart_procrustes_align_gensim(self.model_historical, self.model_modern)

    def fetch_neighbors_cosine(self, word, num_neighbors, use_modern_projected):
        """
        Retrieving a set of nearest neighbors for of the word using cosine similarity metric,
        removing itself (in case of projection) and non-vocabulary words
        :param word: word to center the neighborhood around
        :param num_neighbors: number of nearest neighbors to retrieve
        :param use_modern_projected: toggles between projected modern embeddings (used is 'word' is a neologism)
        and historical embeddings (used if 'word' is a control word)
        :return:
        """
        if use_modern_projected:
            neighbors = self.model_historical.similar_by_vector(self.model_modern_projected.wv[word],
                                                                topn=max(5000, num_neighbors * 20))
        else:
            neighbors = self.model_historical.most_similar(word, topn=max(5000, num_neighbors * 20))

        neighbor_list = []
        for neighbor_word, neighbor_distance in neighbors:
            if neighbor_word in self.vocabulary and neighbor_word != word:
                neighbor_list.append((neighbor_word, neighbor_distance))
            if len(neighbor_list) == num_neighbors:
                return neighbor_list
        print(f"Could only find {len(neighbor_list)} neighbors out of {num_neighbors}")
        return neighbor_list

    def compute_neighborhood_stats_cosine(self, word_pair_dict, outfile_density, outfile_growth):
        """
        Computing density and average frequency growth rate for a range of neighborhoods
        of each neologism and control word
        :param word_pair_dict: neologism - control pair dictionary
        :param outfile_density: file path to output neighborhood densities for each word
        :param outfile_growth: file path to output neighborhood frequency growth rate for each word
        :return:
        """
        mean_neologism_density = [0] * len(COSINE_RADIUS_RANGE)
        mean_control_density = [0] * len(COSINE_RADIUS_RANGE)
        mean_neologism_growth = [0] * len(COSINE_RADIUS_RANGE)
        mean_control_growth = [0] * len(COSINE_RADIUS_RANGE)

        # Counting non-empty neighborhoods for proper averaging of frequency growth
        neologism_nonempty_neighborhood_counts = [0] * len(COSINE_RADIUS_RANGE)
        control_nonempty_neighborhood_counts = [0] * len(COSINE_RADIUS_RANGE)

        d_fout = open(outfile_density, 'w')
        g_fout = open(outfile_growth, 'w')

        for neologism, control in word_pair_dict.items():
            neologism_density = [0] * len(COSINE_RADIUS_RANGE)
            control_density = [0] * len(COSINE_RADIUS_RANGE)
            neologism_growth = [0] * len(COSINE_RADIUS_RANGE)
            control_growth = [0] * len(COSINE_RADIUS_RANGE)

            try:
                neologism_neighbors_5000 = self.fetch_neighbors_cosine(neologism, 5000, use_modern_projected=True)
            except KeyError:
                print(f"{neologism} not found in the modern embedding space vocabulary")
                continue

            try:
                control_neighbors_5000 = self.fetch_neighbors_cosine(control, 5000, use_modern_projected=False)
            except KeyError:
                print(f"{control} not found in the historical embedding space vocabulary")
                continue

            for i, r in enumerate(COSINE_RADIUS_RANGE):
                neologism_neighbors = [w for w, d in neologism_neighbors_5000 if float(d) >= r]
                control_neighbors = [w for w, d in control_neighbors_5000 if float(d) >= r]

                neologism_density[i] = len(neologism_neighbors)
                control_density[i] = len(control_neighbors)
                mean_neologism_density[i] += 1.0 * neologism_density[i] / len(word_pair_dict)
                mean_control_density[i] += 1.0 * control_density[i] / len(word_pair_dict)

                neologism_neighbors_filtered = [w for w in neologism_neighbors if w in self.spearmanr_dict]
                control_neighbors_filtered = [w for w in control_neighbors if w in self.spearmanr_dict]

                if len(neologism_neighbors_filtered) > 0:
                    neologism_growth[i] = np.mean([self.spearmanr_dict[w] for w in neologism_neighbors_filtered])
                    mean_neologism_growth[i] += neologism_growth[i]
                    neologism_nonempty_neighborhood_counts[i] += 1
                else:
                    neologism_growth[i] = "NaN"
                if len(control_neighbors_filtered) > 0:
                    control_growth[i] = np.mean([self.spearmanr_dict[w] for w in control_neighbors_filtered])
                    mean_control_growth[i] += control_growth[i]
                    control_nonempty_neighborhood_counts[i] += 1
                else:
                    control_growth[i] = "NaN"

            d_fout.write("\t".join([neologism] + list(map(str, neologism_density)) +
                                   [control] +list(map(str, control_density))))
            d_fout.write("\n")
            d_fout.flush()
            g_fout.write("\t".join([neologism] + list(map(str, neologism_growth)) +
                                   [control] +list(map(str, control_growth))))
            g_fout.write("\n")
            g_fout.flush()

        for i, r in enumerate(COSINE_RADIUS_RANGE):
            mean_neologism_growth[i] /= neologism_nonempty_neighborhood_counts[i]
            mean_control_growth[i] /= control_nonempty_neighborhood_counts[i]

        print("Neologism neighborhood density: " + "\t".join(["{0:.3f}".format(x) for x in mean_neologism_density]))
        print("Control neighborhood density: ", "\t".join(["{0:.3f}".format(x) for x in mean_control_density]))
        Utils.plot_neighborhood_stats(mean_neologism_density, mean_control_density, "cosine", "density")

        print("Neologism neighborhood frequency growth: " +
              "\t".join(["{0:.3f}".format(x) for x in mean_neologism_growth]))
        print("Control neighborhood frequency growth: " +
              "\t".join(["{0:.3f}".format(x) for x in mean_control_growth]))
        Utils.plot_neighborhood_stats(mean_neologism_growth, mean_control_growth, "cosine", "growth")

        d_fout.close()
        g_fout.close()

    # The following are supporting methods that could be used for additional experiments and visualization
    # They are not integrated in the current verstion of the code

    def compute_neighborhood_stats_euclidean(self, word_pair_dict):
        """
        Computing and plotting mean neighborhood density and frequency growth rate across
        all neologisms and all control words, using Euclidean distance metric instead of cosine similarity
        :param word_pair_dict: neologism - control pair dictionary
        :return:
        """
        historical_vocab = []
        historical_vocab_vectors = np.zeros([len(self.model_historical.wv.vocab), 300], dtype=float)
        for i, word in enumerate(self.model_historical.wv.vocab):
            historical_vocab.append(word)
            historical_vocab_vectors[i, :] = self.model_historical.wv[word]

        neologism_vectors = np.zeros([len(word_pair_dict), 300], dtype=float)
        control_vectors = np.zeros([len(word_pair_dict), 300], dtype=float)

        for i, neologism in enumerate(word_pair_dict.keys()):
            control = word_pair_dict[neologism]
            neologism_vectors[i, :] = self.model_modern_projected.wv[neologism]
            control_vectors[i, :] = self.model_historical.wv[control]

        print("Computing distance matrix for neologisms")
        neologism_dist_matrix = cdist(neologism_vectors, historical_vocab_vectors)
        print("Computing distance matrix for control words...")
        control_dist_matrix = cdist(control_vectors, historical_vocab_vectors)

        mean_neologism_density = [0] * len(EUCLIDEAN_RADIUS_RANGE)
        mean_control_density = [0] * len(EUCLIDEAN_RADIUS_RANGE)
        mean_neologism_growth = [0] * len(EUCLIDEAN_RADIUS_RANGE)
        mean_control_growth = [0] * len(EUCLIDEAN_RADIUS_RANGE)

        for i, r in enumerate(EUCLIDEAN_RADIUS_RANGE):
            # Counting non-empty neighborhoods for proper averaging of frequency growth
            neologism_nonempty_neighborhood_count = 0
            control_nonempty_neighborhood_count = 0

            print(f"Iteration: {i}")
            for j, neologism in enumerate(word_pair_dict.keys()):
                control = word_pair_dict[neologism]
                neologism_neighbors = [historical_vocab[idx] for idx in np.where(neologism_dist_matrix[j] < r)[0]]
                control_neighbors = [historical_vocab[idx] for idx in np.where(control_dist_matrix[j] < r)[0]]

                # Removing the word itself from the list of neighbors
                if neologism in neologism_neighbors:
                    del neologism_neighbors[neologism_neighbors.index(neologism)]
                del control_neighbors[control_neighbors.index(control)]

                mean_neologism_density[i] += 1.0 * len(neologism_neighbors) / len(word_pair_dict)
                mean_control_density[i] += 1.0 * len(control_neighbors) / len(word_pair_dict)

                neologism_neighbors_filtered = [w for w in neologism_neighbors if w in self.spearmanr_dict]
                control_neighbors_filtered = [w for w in control_neighbors if w in self.spearmanr_dict]

                if len(neologism_neighbors_filtered) > 0:
                    mean_neologism_growth[i] += np.mean([self.spearmanr_dict[w] for w in neologism_neighbors_filtered])
                    neologism_nonempty_neighborhood_count += 1
                if len(control_neighbors_filtered) > 0:
                    mean_control_growth[i] += np.mean([self.spearmanr_dict[w] for w in control_neighbors_filtered])
                    control_nonempty_neighborhood_count += 1

            if neologism_nonempty_neighborhood_count > 0:
                mean_neologism_growth[i] /= neologism_nonempty_neighborhood_count
            if control_nonempty_neighborhood_count > 0:
                mean_control_growth[i] /= control_nonempty_neighborhood_count

        print("Neologism neighborhood density: " + '\t'.join([str(d) for d in mean_neologism_density]))
        print("Control neighborhood density: ", '\t'.join([str(x) for x in mean_control_density]))
        Utils.plot_neighborhood_stats(mean_neologism_density, mean_control_density, "euclidean", "density")

        print("Neologism neighborhood frequency growth: " + '\t'.join([str(x) for x in mean_neologism_growth]))
        print("Control neighborhood frequency growth:   " + '\t'.join([str(x) for x in mean_control_growth]))
        Utils.plot_neighborhood_stats(mean_neologism_growth, mean_control_growth, "euclidean", "growth")

    def get_neighborhood_tsne(self, word, radius, use_modern_projected):
        """
        Computing a 2D t-SNE representation of a neighborhood (can be used for visualization)
        :param word: word to center the neighborhood around
        :param radius: cosine similarity threshold to define the neighborhood size
        :param use_modern_projected: toggles between projected modern embeddings (used is 'word' is a neologism)
        and historical embeddings (used if 'word' is a control word)
        :return: list of words in the neighborhoods and t-SNE 2D vector matrix
        """
        try:
            neighbors = self.fetch_neighbors_cosine(word, 100, use_modern_projected=use_modern_projected)
        except KeyError:
            print(f"Error: {word} not found in embeddings")
            exit()

        neighbor_words = [w for w, d in neighbors if float(d) >= radius]
        num_neighbors = len(neighbor_words)

        vectors = np.zeros([num_neighbors+1, 300], dtype=float)
        if use_modern_projected:
            vectors[0, :] = self.model_modern_projected.wv[word]
        else:
            vectors[0, :] = self.model_historical.wv[word]
        for i, neighbor_word in enumerate(neighbor_words):
            vectors[i+1, :] = self.model_historical.wv[neighbor_word]

        tsne_matrix = TSNE(n_components=2).fit_transform(vectors)
        return neighbor_words, tsne_matrix

    def print_nearest_neighbors(self, neologism_list, num_neighbors):
        """
        Printing a list of nearest historical neighbors for each neologism
        :param neologism_list: list of neologisms
        :param num_neighbors: number of neighbors to output per neologism
        :return:
        """
        for neologism in neologism_list:
            try:
                neighbors = self.fetch_neighbors_cosine(neologism, num_neighbors, use_modern_projected=True)
            except KeyError:
                print(f"Neologism {neologism} not found in modern embeddings")
                continue

            print("{0:20}\t{1:10}\t{2:10}\t{3:10}\t{4:10}".format(neologism,
                                                                  *[neighbors[i][1] for i in range(num_neighbors)]))
