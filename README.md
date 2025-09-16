# MetaLog: Generalizable Cross-System Anomaly Detection from Logs with Meta-Learning

## Project Structure
```
├─approaches  # MetaLog main entrance.
├─conf      # Configuration for Drain
├─entities    # Instances for log data and DL model.
├─utils
├─logs        
├─datasets    
├─models      # Attention-based GRU.
├─module      # Anomaly detection modules, including classifier, Attention, etc.
├─outputs           
├─parsers     # Drain parser.
├─preprocessing # Preprocessing code, data loaders and cutters.
├─representations # Log template and sequence representation.
└─util        # Vocab for DL model and some other common utils.
```

## Datasets

We used `2` open-source log datasets, HDFS and BGL. 

| Software System | Description                        | Time Span  | # Messages | Data Size | Link                                                      |
|       ---       |           ----                     |    ----    |    ----    |  ----     |                ---                                        |
| HDFS            | Hadoop distributed file system log | 38.7 hours | 11,175,629 | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)                |
| BGL             | Blue Gene/L supercomputer log      | 214.7 days | 4,747,963  | 708.76MB  | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |


## Environment

Please refer to the `requirements.txt` file for package installation.

**Key Packages:**


PyTorch v1.10.1

python v3.8.3

hdbscan v0.8.27

overrides v6.1.0

scikit-learn v0.24

tqdm

regex

[Drain3](https://github.com/IBM/Drain3)

## Preparation

- **Step 1:** To run `MetaLog` on different log data, create a directory under `datasets` folder HDFS and BGL.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/), and put it under `datasets` folder.

## Run
- Run `approaches/MetaLog.py` (make sure it has proper parameters) for bilateral generalization from HDFS to BGL.
- Run `approaches/MetaLog_BH.py` (make sure it has proper parameters) for bilateral generalization from BGL to HDFS.

## Open5GS Dataset Training and Inference Guide

The repository now ships with clean Open5GS procedure logs in `datasets/open5gs/logs`. This section documents the
end-to-end workflow to train MetaLog on those logs (using HDFS as the meta-training source by default) and then score a
new Open5GS log file.

### Prerequisites

1. **Python environment** – install the dependencies listed in `requirements.txt`.
2. **Word embeddings** – download `glove.6B.300d.txt` from the [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/)
   and place it directly under the repository's `datasets/` directory so the loaders can discover it automatically.
3. **Dataset layout** – the expected structure is:

   ```text
   datasets/
   ├─open5gs/
   │ ├─logs/                             # already populated with clean Open5GS procedures
   │ └─… (generated artifacts will appear here after preprocessing)
   └─glove.6B.300d.txt                   # required embedding file
   ```

   Each `.log` file found under `datasets/open5gs/logs/**` is treated as one block of normal training data. No manual
   labeling is required.

All commands below are intended to be executed from the repository root (`/workspace/MetaLog`).

### Training on Open5GS

Run the following command to meta-train on HDFS and produce Open5GS-specific artifacts without requiring any labeled
Open5GS anomalies:

```bash
python approaches/MetaLog.py \
    --mode train \
    --target_dataset open5gs \
    --source_dataset HDFS \
    --target_zero_label
```

Key notes:

- `--target_dataset open5gs` activates the Open5GS loader and Drain configuration while keeping the shared pipeline
  intact.
- `--target_zero_label` skips the target fine-tuning stage that assumes labeled anomalies, which matches the provided
  Open5GS data.
- The default parser (`IBM`, i.e., Drain3) and training hyper-parameters are reused; override them only if you need to
  experiment.

The run generates parser snapshots, cached embeddings, and GRU checkpoints under `outputs/open5gs/`, following the
standard MetaLog folder layout:

```text
outputs/open5gs/
├─models/MetaLog/open5gs_IBM/model/         # best.pt and last.pt checkpoints
├─results/MetaLog/open5gs_IBM/…             # probabilistic labeling metadata
└─inference/open5gs_IBM/…                   # populated after running inference
```

### Inference on an Open5GS Log File

After training, score any Open5GS-formatted log file with the saved model and Drain state:

```bash
python approaches/MetaLog.py \
    --mode inference \
    --target_dataset open5gs \
    --source_dataset HDFS \
    --inference_file /path/to/open5gs.log \
    --threshold 0.5
```

Replace `/path/to/open5gs.log` with the file you want to analyze (it can be one of the provided logs or any new
Open5GS log captured in the same plain-text format). The `--source_dataset` flag should match the dataset used during
training so the combined vocabulary remains consistent. Adjust `--threshold` if you wish to change the anomaly decision
cut-off; the default of `0.5` replicates the training setting.

The inference report is written to
`outputs/open5gs/inference/open5gs_IBM/layer=2_hidden=100_epoch=10/report.csv` (and a companion `metadata.json`). Each
row in `report.csv` contains the sequence index, the anomaly probability predicted by the trained model, and the binary
decision derived from the supplied threshold.