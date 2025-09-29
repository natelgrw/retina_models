# ReTiNA: Automated LC-MS Retention Time Modeling

ReTiNA is a collection of machine learning models under active development for predicting small molecule retention times in LC-MS workflows.

## ⚗️ The ReTiNA Dataset

The ReTiNA dataset contains:

- 4,358,475 unique molecule–environment combinations, the largest singular LC-MS retention time dataset of its kind to date
- Experimentally measured retention times, in seconds, curated from public datasets, benchmark papers, and literature
- 160 calculated chemical descriptors for 641,647 unique compounds and 6 unique solvents

The dataset is actively expanding with new experimental retention time values from the Coley Research Group at MIT, ensuring it remains a growing resource for optical property prediction.

91 distinct LC-MS setup environments are used in ReTiNA. Each environment consists of:

- Solvent mixtures A and B, consisting of solvents and solvent additives contributing to pH
- The mobile phase gradient used, defined by the percentage of solvent mixture B over time (min)
- The chromatographic technique used - either HILIC or reverse phase 
- LC-MS column dimensions, in terms of column length (mm), internal diameter (mm), and particle size (µm)
- The mobile phase flow rate, measured in mL/min
- The column temperature, measured in degrees Celsius

The full dataset is accessible at this [Hugging Face Repository](https://huggingface.co/datasets/natelgrw/ReTiNA).

ReTiNA is designed for use in:

- Estimating retention times for new compound–environment combinations
- Aiding in peak assignment during LC-MS method development
- Training ML models for retention time prediction under specific conditions

## 📋 Data Sources Used

Detailed information on the data sources comprising the ReTiNA dataset can be found in the the Hugging Face repository linked above.

## ✒️ Citation

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
