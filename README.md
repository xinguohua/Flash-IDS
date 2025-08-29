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

## Code Structure
/
├── atlas_data/           # Stores Atlas-related data files
├── data_files/           # Stores other data files
├── process/              # Core logic and scripts (data processing, model training, reasoning)
│   ├── datahandlers/     # Data loading and preprocessing modules
│   ├── embedders/        # Embedding generation modules
│   ├── match/            # Matching and comparison logic
│   ├── test/             # Test scripts and cases
│   ├── partition.py      # Data partitioning and splitting
│   ├── process_data.py   # Process raw data into usable format
│   ├── reason_graph.py   # Graph-based reasoning
│   ├── reasoning.py      # General reasoning logic
│   ├── test.py           # Test runner
│   ├── test_local_chatgpt.py # Local ChatGPT testing script
│   ├── train.py          # Model training script
│   ├── type_enum.py      # Enum definitions for types
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies list



