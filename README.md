This repository contains the code and supplementary materials for the [paper](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1148&context=scil):
```
Where New Words Are Born: Distributional Semantic Analysis of Neologisms and Their Semantic Neighborhoods
Maria Ryskina, Ella Rabinovich, Taylor Berg-Kirkpatrick, David R. Mortensen, Yulia Tsvetkov
SCiL 2020
```

Please contact mryskina@cs.cmu.edu for any questions.

Embedding alignment code in `projection.py` is based on [Ryan Heuser's Gensim port](https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf) of William Hamilton's alignment code in [HistWords](https://github.com/williamleif/histwords).

## Data

This code uses the [COHA](https://www.english-corpora.org/coha/) and [COCA](https://www.english-corpora.org/coca/) corpora in plain text format. The corpora need to be downloaded from https://www.corpusdata.org/.

## Usage

Code to train the historical and modern embeddings using [Gensim](https://radimrehurek.com/gensim/):
```
python train_w2v.py <coha_path> historical
python train_w2v.py <coca_path> modern
```
where `<coha_path>` and `<coca_path>` need to be replaced with paths to COHA and COCA top-level text directories respectively. Trained embedding models will be saved into the `models` directory. 

Pretrained embeddings will be available shortly.

Code to reproduce the main analysis:
```
python main.py <coha_path> <coca_path> [--seed <seed>] [--stable]
```
where:
* `--seed` is an optional argument specifiying a random seed used to randomize control set selection
* `--stable` flag switches between stable and relaxed control sets

The MATLAB script for fitting the generalized linear model (GLM) can be found in `glm.m`.

## Files

* `vocabulary.txt` contains a vocabulary of nouns extracted from [Wikicorpus](https://www.cs.upc.edu/~nlp/wikicorpus/)
* `neologisms.txt` is a list of neologisms automatically extracted by our code
* `freq_growth.tsv` contains the frequency growth rates (Spearman's correlation coefficients and p-values) for all vocabulary words
* `pairs.{stable|relaxed}.tsv` is a list of neologism-control pairs for stable and relaxed control sets respectively
* `density.{stable|relaxed}.tsv` and `growth.{stable|relaxed}.tsv` display neighborhood density and average frequency growth rate for a range of neighborhood sizes for each neologism and control word
* `glm.{stable|relaxed}.tsv` is a reformatting of the density and growth data to be used for GLM fitting
* `Supplementary.xlsx` contains detailed results of the regression analysis and collinearity tests and nearest historical neighbors for all neologisms

## Dependencies

Core dependencies:
  * Python >= 3.5
  * SciPy >= 1.0.1
  * Gensim >= 3.7.0
  * NLTK
  * MATLAB (for GLM analysis only)

## Reference
 ```
 @article{ryskina2020where,
  title={Where New Words Are Born: Distributional Semantic Analysis of Neologisms and Their Semantic Neighborhoods},
  author={Ryskina, Maria and Rabinovich, Ella and Berg-Kirkpatrick, Taylor and Mortensen, David R. and Tsvetkov, Yulia},
  journal={Proceedings of the Society for Computation in Linguistics},
  volume={3},
  number={1},
  pages={43--52},
  year={2020}
}
 ```
