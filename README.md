# AttackWeaver 
Welcome to the AttackWeaver repository. Here, we offer the implementation details of the method introduced in our research paper titled "Weaving Fragments to Chains: Match then Reason through GNN-LLM Synergy for Threat Hunting". 

## Installation
`requirements.txt` file is provided with all required dependencies and their specific versions. Install them using the following command:
```bash
pip install -r requirements.txt
```

## Datasets
AttackWeaver is evaluated on open-source datasets from Darpa and the research community. You can access these datasets using the following links.

### Darpa E3
```bash
https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md
```

### Darpa E5
```bash
https://github.com/darpa-i2o/Transparent-Computing/blob/master/README.md
```

### ATLAS
```bash
https://github.com/purseclab/ATLAS/tree/main/raw_logs
```

### Darpa OpTC
```bash
https://github.com/FiveDirections/OpTC-data
```

### APTData Share Link 
File Description:
This file, named “aptdata”, was shared via Baidu Cloud. It contains data related to APT (Advanced Persistent Threat) analysis and labeling, including information used for threat behavior annotation and model evaluation.

Access Link:
Baidu Cloud Share: https://pan.baidu.com/s/1AtDR5NuMHTnzAcmLJC_c8A?pwd=imgn￼
Extraction Code: imgn


## Code Structure
```
/                           # Root directory
├── atlas_data/             # Stores Atlas-related data files
├── data_files/             # Stores E3 data files
├── data_files5/            # Stores E5 data files
├── data_files_optc/        # Stores optc data files
├── process/                # Core logic and scripts (data processing, model training, reasoning)
│   ├── preprocess/          # Data preprocessing scripts and utilities
│   │   ├── process_data.py      # Specialized preprocessing for E3 data
│   │   ├── process_data_E5.py   # Specialized preprocessing for E5 data
│   │   └── process_data_optc.py # Specialized preprocessing for OPTC data
│   ├── datahandlers/       # Data loading and preprocessing modules
│   │   ├── atlas_handler.py            # Handler for ATLAS dataset
│   │   ├── darpa_handler.py            # Handler for DARPA dataset (general version)
│   │   ├── optc_handler.py             # Handler for OPTC dataset
│   │   └── darpa_handler5.py           # Handler for DARPA dataset (phase-5 variant)
│   ├── embedders/          # Embedding generation modules
│   ├── match/              # Matching and comparison logic
│   ├── test/               # Test scripts and cases
│   ├── partition.py        # Data partitioning and splitting
│   ├── reason_graph.py     # Graph-based reasoning
│   ├── reasoning.py        # General reasoning logic
│   ├── test.py             # Test runner
│   ├── test_local_chatgpt.py # Local ChatGPT testing script
│   ├── train.py            # Model training script
│   └── type_enum.py        # Enum definitions for types
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies list
```


