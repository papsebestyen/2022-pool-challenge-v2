# 2022-pool-challenge-v2

## Setup

After forking this repo and cloning your own fork, run these commands in the terminal:

```
>> cd <path-to-folder>2022-pool-challenge-v2
>> pip install -r requirements.txt
>> inv get-test-pack
```

Then open a python environment in the same folder and unzip `pack.zip`:

```
import zipfile

with zipfile.ZipFile("pack.zip") as zip_path:
    zip_path.extractall(".")
```

## Task

A valid solution is presented as a directory.

It must contain one [yaml](https://en.wikipedia.org/wiki/YAML) file: `conf.yaml`,
plus any other files that are accessed by the commands defined in this yaml.

In the yaml file, on the top level, there should be 4 keys with the following values:

* `image` (optional): The docker image base, where the solution can run. (Example: `python:3.8`)
* `setup-command` (optional): Sets up the environment, where the other commands can run. (Example: `pip install -r requirements.txt`)
* `etl-command` (optional): It runs assuming that a `data.csv` file containing the experimental data is present in the root of the solution. It can do anything with the data to prepare it for the process command. (Example: `python preprocess.py`)
* `process-command` (**mandatory**): It runs assuming that an `input.json` file containing the input locations is present in the solution root. Your task is to make this command write out the answers to the queries found in the inputs into an `output.json` file in the root of the solution, as fast as possible. (Example: `python main.py`)

Your solutions will be evaluated based on:
* base speed
* scaling with the size of the query
* scaling with the size of the data
