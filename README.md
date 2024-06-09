# A Joint Embedding Learning Model for Fine-Grained Entity Type Error Detection in Noisy Knowledge Graphs

This is the code of A Joint Embedding Learning Model for Fine-Grained Entity Type Error Detection in Noisy Knowledge Graphs.

## Dependencies

- pytorch==2.3.0(cuda11.8)
- transformers==4.11.3
- python==3.9
- dgl==1.1.2
- tqdm==4.64.1 
- rdflib==7.0.0 

## Running
### DFTFM

Run DFTFM/text.py. 

The parameters in DFTFM/GlobalValue.py including training epochs, max text length, batchsize, embedding dimension, error rate, which can be modified.

After training, it output entity_features.json and type_features.json two files in BFTFM_output folder.


### LTEM

You can run the command in LTEM/script/script.sh.
We list three different KGE methods here.

The parameters in LTEM/GlobalValue.py can also be modified. You can change the argument settings such as: 
'--epoch': training epochs , 
'--batch': batch size,
'--errorrate': the error rate of entity-type pairs and so on. 

Note that '--init_dim' and '--gcn_dim' must be the same as the dimensions in DFTFM. 

After training, it output entities.json, relations.json, types.json, weights.txt files in LTEM_output folder, which can be applied for testing in test.py.




          