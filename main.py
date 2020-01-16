from extract_word_stats import WordStatsExtractor
from extract_neighborhood_stats import NeighborhoodStatsExtractor
import argparse

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("coha_path", type=str,
                        help="Path to the COHA text directory (containing the decade subdirectories")
    parser.add_argument("coca_path", type=str,
                        help="Path to the COCA text directory (containing the genre subdirectories)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for randomizing control sets")
    parser.add_argument("--stable", action='store_true', help="Turn on stability constraint for the control set words")
    return parser.parse_args()


def main(params):
    seed = params.seed
    stability_constraint = params.stable

    ws = WordStatsExtractor(params.coha_path, params.coca_path)

    # Collecting vocabulary

    vocab_path = "files/vocabulary.txt"
    vocab = Utils.read_vocabulary(vocab_path)
    print(f"Loaded Wikicorpus vocabulary with {len(vocab)} nouns")

    # Extracting word frequencies

    print("Extracting historical and modern word frequencies...")
    ws.extract_frequencies(vocab, data_split="historical")
    ws.extract_frequencies(vocab, data_split="modern")
    print("Done.")

    # Extracting neologisms

    print("Extracting a list of neologisms...")
    neologism_filename = "files/neologisms.txt"
    neologism_list = ws.extract_neologisms(neologism_filename)
    print("Done.")

    # Estimating frequency growth trends for all nouns in the vocabulary

    growth_filename = "files/freq_growth.tsv"
    print("Estimating word frequency growth trends...")
    frequency_growth_dict = ws.extract_frequency_growth(vocab, growth_filename)
    print("Done.")

    # Pairing neologisms with control

    pair_filename = \
        f"files/pairs.{'stable' if stability_constraint else 'relaxed'}" \
        f"{'.seed' + str(seed) if seed is not None else ''}.tsv"
    print(f"Pairing neologisms with {'stable' if stability_constraint else 'relaxed'} control words...")
    neologism_control_pairs = ws.pair_neologisms_with_controls(frequency_growth_dict, neologism_list, pair_filename,
                                     stability_constraint=stability_constraint, seed=None)
    print("Done.")

    print("Loading and aligning embedding models...")
    historical_model_file_path = "models/historical.w2v.bin"
    modern_model_file_path = "models/modern.w2v.bin"
    ns = NeighborhoodStatsExtractor(historical_model_file_path, modern_model_file_path, vocab, frequency_growth_dict)
    print("Done.")

    # Computing neighborhood density and average frequency growth rate

    density_filename = f"files/density.{'stable' if stability_constraint else 'relaxed'}" \
                       f"{'.seed' + str(seed) if seed is not None else ''}.tsv"
    growth_filename = f"files/growth.{'stable' if stability_constraint else 'relaxed'}" \
                      f"{'.seed' + str(seed) if seed is not None else ''}.tsv"

    print("Estimating neighborhood density and average frequency growth rates...")
    ns.compute_neighborhood_stats_cosine(neologism_control_pairs, density_filename, growth_filename)
    print("Done.")

    # Reformatting output to use in GLM

    glm_filename = f"files/glm.{'stable' if stability_constraint else 'relaxed'}" \
                   f"{'.seed' + str(seed) if seed is not None else ''}.csv"

    print("Reformatting feature files for inputting to GLM script...")
    Utils.reformat_feats_for_glm(density_filename, growth_filename, COSINE_RADIUS_RANGE, glm_filename)
    print("Done.")

# ----------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)
