# Dependencies

## Data

### PeerRead:

Collect the PeerRead dataset from https://github.com/allenai/PeerRead:  

```bash
mkdir data
cd data/
git clone https://github.com/allenai/PeerRead.git
```

## Libraries

The python libraries can be found in the `requirements.txt` file 
and can be installed by:

```bash
pip install -r requirements.txt
```

# Executing the files

The files can be executed by running, for instance, the following:
```bash
python src/basic_GRU_Classifier.py
```
Those scrips need to run from the ROOT directory of the project.