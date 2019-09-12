> Deprecated: See the [current version](https://github.com/erayyildiz/neural-morphological-disambiguation) which is based on PyTorch.

# Neural-Morphological-Disambiguation-for-Turkish

This project is dynet implementation of morphological disambiguation method proposed in [1]

Full context model is one of the methods proposed in the study and perform best for Turkish morphological disambiguation among four proposed methods.
All the methods are based on character level bidirectional LSTM networks and Full Context Model represent the context applying LSTMs on surface words around the target word needs to be disambiguated.
Here, we provide an implementation of this method written in dynet. 
We also provide train and test sets for Turkish morphological disambiguation.
The disambiguation model can be trained using this dynet script. 
Model save and load functions are also provided so that users are able to use this tool as a Turkish morphological disambiguator.


This script is developed in a project about joint dependecy parsing and morphological disambiguation and willbe further developed. 
A better organized morphological disambiguation tool with much more functionality will be provided soon.

Just run them Neural_morph_disambiguator.py to start training. 
Values of default parameters are same as proposed values in the paper.

For the details about the method, please see the Shen's study.

accuracy on test set:  0.966707477868  ambiguous accuracy on test:  0.93011416992

[1] Shen,et. al. 2017. The Role of Context in Neural Morphological Disambiguation COLING 2016


## Installation
To install the progrm you need a machine with a Python 3 or higher.
To install python3, run the following program if your machine has linux based OS.

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
```

Then you should install python package manager pip:

```bash
sudo apt install python3-pip
```

After pip is installed you should install python packages which the project requires:

```bash
pip install -f REQUIREMENTS.txt
```
Then create a folder named `models` in project root directory so that the trained model can be saved into this directory.
```bash
mkdir models
```

Now installation is completed, you can run the program.

## How to  run

For training you can run the following python script:

```python
from neural_morph_disambiguation_dynet import MorphologicalDisambiguator
disambiguator = MorphologicalDisambiguator(train_from_scratch=True, char_representation_len=100, word_lstm_rep_len=200, 
                                           train_data_path="data/data.train.txt",
                                           test_data_paths=["data/data.test.txt","data/test.merge"],
                                           model_file_name='my_model')
```

These codes will start training on given datasets and you can track the training status from console.
When the training is completed, the model will be saved into `models` path with name '*my_model*'

To load and use a trained model use the codes below:

```python
from neural_morph_disambiguation_dynet import MorphologicalDisambiguator
disambiguator = MorphologicalDisambiguator.create_from_existed_model("model-22.10.2017")
```
