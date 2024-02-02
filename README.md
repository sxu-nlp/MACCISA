##  MACCISA-pytorch

##  Introduction

We designed the MACCISA model, it contains three parts:  
1. We first obtain multimodal semantic matrices utilizing adapted macromodels.

2. We use multiple adapted models to obtain common-sense knowledge representations.

3. We use the multi-head attention mechanism and the adapter layer to merge and learn from the obtained multimodal semantic matrices.

4. Due to the large size of the dataset and multimodal commonsense knowledge graph files, we will release the web address after the anonymous review period.

## Environment Requirement

`pip install -r requirements.txt`
Also, please make sure you have the Chatglm-6b model in your runtime environment.

## Dataset

We provide a processed English dataset and a Chinese one.
`data_en.xlsx/data_ch.xlsx`
The corresponding node number sequences for the one-hop and two-hop dependencies of the relevant desired knowledge graph nodes in the dataset sentences are also provided.
`one-hop_ch.xlsx/one-hop_en.xlsx` and `two-hop_ch.xlsx/two-hop_en.xlsx`



## Modification of the Chatglm-6b model
Firstly replace `.. /ChatGLM-6B-main/chatglm-6b/modelling_chatglm.py` with the chat function in `modelling_chatglm_c.py` that we provided. Also, replace the utils.py file where **the generate function is located** in modeling_chatglm.py with the `utils_C.py` file we provided.

## An example to  run the model
Using Chinese dataset and Chinese one-hop data:
* Modify dataset path
Change `dataset_path、related_nodes_path` in `model.py`

* Use the graph node vector file and image vector file of the corresponding language version
Change `node_vectors_path、picture_vectors_path` in `model.py` using `ch_embedding.pkl、pictures_embedding_ch.pkl` we provided.
(*These two files are vector representations learned by the common sense knowledge base model and image processing model mentioned in the paper.*)

 * Modify node demarcation number
    Change `demarcation_num` in `model.py`.If you are using the Chinese dataset, you should set it to **11386**. If you are using the English dataset, you should set it to **30432**.
	(*This parameter represents the number of word nodes and distinguishes word nodes and picture nodes.*)
	
	
 * Run the model
 After making the above modifications, you can execute `python ../model.py` through the command line to train the model.




