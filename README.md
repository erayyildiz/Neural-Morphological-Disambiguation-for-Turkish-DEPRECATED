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

[1] Shen,et. al. 2017. The Role of Context in Neural Morphological Disambiguation COLING 2016