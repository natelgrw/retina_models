# ReTiNA: Automated LC-MS Retention Time Modeling

ReTiNA is a collection of machine learning models under active development for predicting small molecule retention times in LC-MS workflows.

Current Version: **1.0.0**

Active ReTiNA prediction models are accessible at this [Hugging Face Model Repository](https://huggingface.co/natelgrw/ReTiNA-Models). Depreciated models are available upon request at `natelgrw.tech@gmail.com`. 

This repository contains scripts for model training and processing, along with model performance metrics.

## 🌀 Models

For retention time prediction, we recommend that you use **ReTiNA_XGB1**, the model with the greatest overall prediction accuracy.

| Model | Architecture | Overall RMSE (s) | Overall MAE (s) | Overall R<sup>2</sup> | Status |
|-----|-----|-----|-----|-----|-----|
| **ReTiNA_XGB1** | XGBoost | 182.81 | 119.30 | 0.659 | Active |
| **ReTiNA_MLP1** | PyTorch Residual MLP | 202.67 | 141.79 | 0.516 | Active |

All models were evaluated across rigorous scaffold, cluster, and method splits.

## ⚗️ The ReTiNA Dataset

The ReTiNA dataset contains:

- 119,039 unique molecule–environment combinations, the largest singular LC-MS retention time dataset of its kind to date
- Experimentally measured retention times, in seconds, curated from public datasets, benchmark papers, and literature
- Calculated chemical descriptors for 105,809 unique compounds, 6 unique solvents, and 8 unique additives

The dataset is actively expanding with new experimental retention time values from the Coley Research Group at MIT, ensuring it remains a growing resource for optical property prediction.

73 distinct LC-MS setup environments are used in ReTiNA. Each environment consists of:

- Solvent mixtures A and B, consisting of solvents and solvent additives contributing to pH
- The mobile phase gradient used, defined by the percentage of solvent mixture B over time (min)
- The chromatographic technique used - either HILIC or reverse phase 
- LC-MS column dimensions, in terms of column length (mm), internal diameter (mm), and particle size (µm)
- The mobile phase flow rate, measured in mL/min
- The column temperature, measured in degrees Celsius

The full dataset is accessible at this [Hugging Face Repository](https://huggingface.co/datasets/natelgrw/ReTiNA).

The ReTiNA dataset is designed for use in:

- Estimating retention times for new compound–environment combinations
- Aiding in peak assignment during LC-MS method development
- Training ML models for retention time prediction under specific conditions

## 📋 Data Sources Used

Detailed information on the data sources comprising the ReTiNA dataset can be found in the the Hugging Face repository linked above.

## ✒️ Citation

If you use a ReTiNA prediction model in a project, please cite the following:

```
@modelcollection{retinamodels,
  title={ReTiNA-Models: Machine Learning Models for LC-MS Retention Time Prediction},
  author={Leung, Nathan},
  institution={Coley Research Group @ MIT}
  year={2025},
  howpublished={\url{https://huggingface.co/natelgrw/ReTiNA-Models}},
}
```

If you use the ReTiNA dataset in a project, please cite the following:

```
@dataset{natelgrwretinadataset,
  title={ReTiNA: A Benchmark Dataset for LC-MS Retention Time Modeling},
  author={Leung, Nathan},
  institution={Coley Research Group @ MIT}
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/natelgrw/ReTiNA}}
}
```
