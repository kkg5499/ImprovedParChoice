# ParChoice

This repository contains code for producing style transfer in text with four techniques described in:
[Tommi Gröndahl and N. Asokan. "Effective writing style transfer via combinatorial paraphrasing"](https://content.sciendo.com/view/journals/popets/2020/4/article-p175.xml).

This repository contains code for the *stand-alone* parts of ParChoice. The paper additionally reports comparative experiments using external GitHub repositories:
- [Language style transfer (cross-aligned autoencoder)](https://github.com/shentianxiao/language-style-transfer)
- [Style-transfer-through-back-translation](https://github.com/shrimai/Style-Transfer-Through-Back-Translation)
- [A4NT](https://github.com/rakshithShetty/A4NT-author-masking/blob/master/README.md)
- [PAN 2016 AuthorObfuscation](https://bitbucket.org/pan2016authorobfuscation/authorobfuscation/src/master/)
- [Mutant-X](https://github.com/asad1996172/Mutant-X/)

For more information, please contact the paper authors.

If you use this code in scientific work, please cite:
```
@article { Effectivewritingstyletransferviacombinatorialparaphrasing,
      author = "Tommi Gröndahl and N. Asokan",
      title = "Effective writing style transfer via combinatorial paraphrasing",
      journal = "Proceedings on Privacy Enhancing Technologies",
      year = "01 Oct. 2020",
      publisher = "Sciendo",
      address = "Berlin",
      volume = "2020",
      number = "4",
      doi = "https://doi.org/10.2478/popets-2020-0068",
      pages=      "175 - 195",
      url = "https://content.sciendo.com/view/journals/popets/2020/4/article-p175.xml"
}
```

## Setup

We recommend running the code on an Anaconda virtual environment. If you do not have Anaconda installed, see the link for installation instructions:
https://docs.anaconda.com/anaconda/install/

The file env.yml can be used to create an Anaconda environment (called "parchoice") with the required dependencies as follows:

```bash
conda env create -f env.yml
```

(The virtual environment with all dependencies requires 2.8 GB of memory.)

If your Anaconda version >4.4, activate the environment as follows:

```bash
conda activate parchoice
```

If you Anaconda version <4.4, activate the environment as follows:
```bash
source activate parchoice
```


## Running a test experiment

To run a test experiment, run:
```bash
bash example.sh
```

The file example.sh downloads the required Python dependencies, and runs an example experiment on the data found in the "data" folder. It trains a logistic regression classifier on the files "alice_train.txt" and "bob_train.txt", to classify between texts written by "alice" or "bob". It then uses ParChoice to transform "alice_test.txt" and closer to the style of "bob", as determined by the trained classifier.

Resulting transformations are stored in the "results" folder under the name "alice_test_transf_ppdb_wn_typos". The name contains all techniques used in the transformation (ppdb, wn, and typos).

The folders "ppdb", "inflections", and "symspell" contain additional material required for running style transfer; see separate README and LICENCE files in each folder.


## Running a custom experiment

Run the style transfer pipeline as follows:

```bash
python style_transfer/main.py --src <path_to_source_file> --src_train <path_to_source_training_corpus> --tgt_train <path_to_target_training_corpus> --use_ppdb --use_wordnet --use_typos --spell_check
```

All source and target files should be .txt-files with one sentence per row.

The classifier is trained on the source training corpus and target training corpus. The style transfer is conducted on the source file.

All arguments except "src" are optional. Running without "src_train" or "tgt_train" conducts random paraphrasing.

Paraphrase algorithms (simple transformations, WordNet, PPDB, typos) are defined in style_transfer/paraphrases.py.

Simple transformations (see paper) are always generated, even in the absence of any optional parameters.


## Copyright and licence

Copyright (C) 2020 Secure Systems Group, Aalto University and University of Waterloo
License: see LICENSE.txt files in each subfolder.
