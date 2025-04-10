# Future Pipeline

This code stems from [WARC-DL](https://github.com/webis-de/WARC-DL) mostly.

## Usage

An appropriate config file must be created as `./config.ini`. A template can be found [here](https://github.com/webis-de/WARC-DL/blob/master/config-template.ini).

Our Docker Image must be used. It was derived from the [WARC-DL Docker Image](https://github.com/webis-de/WARC-DL/pkgs/container/WARC-DL).

### Using the Docker Image

To use the Docker Image execute the following command from within this directory:

```
HADOOP_USER_NAME=$USER \
srun --export=ALL --pty --mem=32g \
--container-name=warc-dl --container-image=<path-to-sqsh-file> \
--container-mounts=/mnt/ceph:/mnt/ceph --container-writable \
--gres=gpu:1g.5gb \
bash -c "cd $(pwd) && PYTHONPATH=. HADOOP_CONF_DIR='./hadoop/' python3 <path-to-pipeline>"
```

Don't forget to replace `<path-to-sqsh-file>` and `<path-to-pipeline>` with actual paths. Our pipelines are in the [`pipelines`](pipelines/) directory. Examples from WARC-DL can be found in [`examples`](examples/)

### [`FutureRegexPipeline`](pipelines/future_regex_pipeline.py)

This pipeline extracts sentences from the webis webarchive based on regexes. We used this Pipeline to generate data for training our models.

### [`RnnPredictionPipeline`](pipelines/rnn_prediction_pipeline.py)

This pipeline can be used to predict sentences with our RNN / LSTM model. That model was created to serve as a baseline.

### [`DistilBertPipeline`](pipelines/distilbert_pipeline.py)

This pipeline can be used to predict sentences with our finetuned Distilbert model. There is also the [`DistilBertPredictionPipeline`](distilbert_prediction_pipeline.py) which should also predict sentences with the Distilbert model, but it throws errors from time to time. 

In this pipeline we changed the prediction to predict on numpy batches via the `model.predict_on_batch` keras function when iterating over the dataset instead of calling the model directly and mapping the dataset with the model call. 

We haven't seen errors on this pipeline so far.

### [`PassthroughPipeline`](pipelines/passthrough_pipeline.py)

This pipeline was created to be able to extract sentences when the DistilBertPipeline still threw errors. We predicted those sentences afterwards. It should be obsolete by now.

### Other Pipelines

The other pipelines are abstract pipelines that are imported from the pipelines mentioned above.
