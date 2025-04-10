# Datasets

Our datasets are saved in a dvc (data version control). If there are only .dvc files present here, the actual datasets 
can be pulled by calling `dvc pull`. Our three main datasets are `raw_futurist_data`, `raw_nineties_data` and 
`raw_regex_pipeline_data`. There are also 200 examples in `evaluation_data.jsonl` that were manually crafted to evaluate 
the models qualitatively.

## Versions

For each of the main datasets there is a splitted-jsonl version which holds the original data but splitted into train, 
test and validation data.

There is also a splitted-embeddings version which holds the embedded CLS-Token from a DistilBERT model for each line of 
the splitted-jsonl version. Those datasets were created to speed up the training of a classification head for the 
DistilBERT model.