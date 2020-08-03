# Dependencies

## Data

### PeerRead:

Collect the PeerRead dataset from https://github.com/allenai/PeerRead:  

```bash
mkdir data
cd data/
git clone https://github.com/allenai/PeerRead.git
```

### Embeddings

To save the embeddings to a file sun the following:
```bash
python src/DataLoader.py right mid
```

This will produce both the embeddings when truncating
the end of the review as well as the middle. 
This file must be executed from the root of the project.


## Libraries

The python libraries can be found in the `requirements.txt` file 
and can be installed by:

```bash
pip install -r requirements.txt
```

# Executing the files

The files can be executed by running, for instance, 
the following:

```bash
python src/basic_GRU_Classifier.py
```
Those scrips need to be executed from the ROOT directory of 
the project.

# Get Features from PeerRead

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

```