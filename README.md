# ML4RG-group-17: lncRNA-Protein Interaction Prediction

A machine learning research project focused on predicting interactions between long non-coding RNAs (lncRNAs) and proteins using Graph Neural Networks (GNNs). This project combines data from the LncTarD database and STRING protein-protein interaction database to build heterogeneous graphs for link prediction tasks.

## Project Overview

This project implements various Graph Neural Network architectures to predict potential interactions between lncRNAs and proteins, which is crucial for understanding gene regulatory mechanisms and disease pathways. The models use BioBERT embeddings for node features and explore different GNN architectures including GraphSAGE, Graph Attention Networks (GAT), GENeralized Graph Convolution, and Graph Transformers.

### Key Features

- **Heterogeneous Graph Construction**: Combines lncRNA-gene interactions from LncTarD with protein-protein interactions from STRING
- **Multiple GNN Architectures**: GraphSAGE, GAT, GATv2, GENConv, Graph Transformers, and Enhanced HAN
- **Advanced Node Embeddings**: BioBERT embeddings, and node and edge types and features.
- **Link Prediction**: Binary classification to predict potential lncRNA-protein interactions
- **Performance Evaluation**: Uses AUC-ROC scores for model evaluation

## Getting Started

### Prerequisites

- Python 3.12 or higher

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── data/
│   ├── biobert_embeddings/           # BioBERT embeddings in various dimensions
│   ├── datasets/                     # Raw and processed datasets
├── graphs/                           # Preprocessed graph data files
├── models/                           # Model implementations and experiments
├── node_embeddings/                  # Final result: node embeddings
├── scripts/                          # Data preprocessing and graph creation
└── requirements.txt                  # Python dependencies
```

## Dataset

The project uses three main data sources:

1. **LncTarD Database**: Contains lncRNA-gene regulatory relationships with disease associations
2. **STRING Database**: Protein-protein interaction network with confidence scores (filtered at ≥700)
3. **Dataset from "Genome-scale pan-cancer interrogation of lncRNA dependencies using CasRx"**

### Data Processing

- **LncTarD**: Filtered to keep only lncRNA regulators, mapped to gene symbols
- **STRING**: High-confidence interactions (score ≥700), ENSP IDs mapped to gene names
- **Integration**: Merged datasets to create a comprehensive interaction network
- **CasRx**: Merged with the rows that could be mapped to unique ENSG IDs (using gProfiler), which were not mapped in other rows too, and for those that were also contained in lncTarD

## Models

### Implemented Architectures

1. **GraphSAGE**: Scalable graph neural network with neighbor sampling
   - Baseline implementation
   - Node type features
   - Node degree features  
   - Combined node features

2. **Graph Attention Networks (GAT/GATv2)**: Attention-based message passing
   - Homogeneous and heterogeneous variants
   - Multiple embedding dimensions (50D, 128D)

3. **Graph Transformers**: Transformer architecture adapted for graphs

4. **Enhanced HAN**: Heterogeneous Attention Network with improvements

5. **GENConv**: GENeralized Graph Convolution
   - For heterogeneous data
   - With multiple embedding dimensions (50D, 128D)

## Usage

### Data Preprocessing - Creating the graph

0. Create BioBERT embeddings with the desired dimensions:
   ```bash
   jupyter notebook scripts/pca_biobert_embeddings.ipynb
   ```
  
1. Create the graph as homogeneous "Data" object, but containing heterogeneous data:
   ```bash
   jupyter notebook scripts/create_homogeneous_graph.ipynb
   ```

2. Convert the homogeneous "Data" object to "HeteroData":
   ```bash
   jupyter notebook scripts/convert_homo_to_hetero_data.ipynb
   ```

### Training Models

Navigate to the `models/` directory. The final models which are used for creating the node embeddings can be found in:

```bash
# Using BioBERT embeddings of reduced dimensionality to 50 to create graph, which is then used to train the GNN (HeteroConv using GENConv layers) on link prediction, and then applied to create node embeddings of dimensionality 50 for all nodes in the graph: 
jupyter notebook models/link_prediction_hetero_gen_50d.ipynb

# Same as above, just using BioBERT and output embeddings of size 128:
jupyter notebook models/link_prediction_hetero_gen_128d.ipynb
```

### Evaluation

Models output AUC-ROC scores for validation and test sets. Training logs show loss progression and final performance metrics.

## Technologies

- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **BioBERT**: Pre-trained biomedical language model for embeddings
- **NetworkX**: Graph analysis and manipulation
- **scikit-learn**: Machine learning utilities and metrics
- **pandas/numpy**: Data manipulation and numerical computing

## Research Context

This project is part of ML4RG (Machine Learning for Regulatory Genomics) research, focusing on understanding regulatory relationships between lncRNAs and proteins. The work contributes to computational biology by applying graph-based machine learning techniques to predict previously unknown lncRNA-protein interactions, which could help identify new therapeutic targets and understand disease mechanisms.
