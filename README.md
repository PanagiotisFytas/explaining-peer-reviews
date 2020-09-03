# Explaing Peer Reviews

Using Machine Learning, Explainable AI and Causal Inference to determine "What Make a Scientific Paper Be Accepted for Publication". This work is part of my thesis for the degree of MSc Advanced Computing from Imperial College London.

## Data Dependencies

### PeerRead:

Collect the PeerRead dataset from https://github.com/allenai/PeerRead:  

```bash
mkdir data
cd data/
git clone https://github.com/allenai/PeerRead.git
export DATA=$(pwd)
```

Be sure to export the path to directory where PeerRead is in, as specified above. This is essential
for reading the data from PeerRead.

<!-- ### Embeddings

To save the embeddings to a file sun the following:
```bash
python src/DataLoader.py right mid
```

This will produce both the embeddings when truncating
the end of the review as well as the middle. 
This file must be executed from the root of the project. -->


## Libraries Dependencies

The python libraries can be found in the `requirements.txt` file 
and can be installed by:

```bash
pip install -r requirements.txt
```

## Executing the Files

The files can be executed by running, for instance, 
the following:

```bash
python src/basic_GRU_Classifier.py
```
Those scrips need to be executed from the ROOT directory of 
the project.

## Overview of Main Files

- **DataLoader.py** :
- **helper_functions.py** :
- **models.py** :
- **abstract_classifier.py** :
- **length_hist.py** :
- **logistic_regression_curves_per_review.py** :
- 

<!-- # Get Features from PeerRead

Run on Python:

```Python
import nltk
nltk.download('punkt')
```

Run on bash:

```bash
cd $DATA/PeerRead/code/accept_classify/
./run_featurize_classify.sh
```

Run in order to produce grammatical errors:
    
```
from DataLoader import DataLoader
d = DataLoader('cpu')
d.read_labels().shape
feat = d.read_handcrafted_features()
perr, aerr, pwor, awor = d.read_errors()

``` -->