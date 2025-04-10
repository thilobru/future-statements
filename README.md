# Docker Images

We have created three different Docker Images to run our code:

- `registry.gitlab.com/bdalt/future-statements/pipelines`
- `registry.gitlab.com/bdalt/future-statements/training`
- `registry.gitlab.com/bdalt/future-statements/sentiments`

The first one is to run our pipelines, the second one is for training our models and the last one 
is to perform a sentiment analysis on the final dataset. The pipeline image can also be used for 
model training.

To import the images one must first create credentials for the gitlab container registry:

```bash
echo 'machine registry.gitlab.com login <login> password <password>' >> ~/.config/enroot/.credentials
```

Each image can be imported via enroot:

```bash
srun enroot import -o <sqsh_file_name> docker://<selected_image>
```

For example:

```bash
srun --mem=64G enroot import -o sentiments.sqsh docker://registry.gitlab.com/bdalt/future-statements/sentiments:latest
```

# dvc

We used dvc to version our datatsets. The datasets can be imported via `dvc pull`.

# Programs

All our programs can be found under [`src/`](src/). Pipeline code can be found under [`src/future-pipeline`](src/future-pipeline). 
Code to train our models lies under [`src/ml/`](src/ml). 
