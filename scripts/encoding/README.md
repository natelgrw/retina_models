# 🔒 ReTiNA Encoder Specification

The lastest version of ReTiNA encodes 292 features per datapoint according to specifications. Models are trained with this encoding to predict compound retention time.

## Feature Encoding Breakdown

| Component | Features | Description |
|-----------|----------|-------------|
| **Compound** | 156 | All descriptors from `comp_descriptors.csv` except Morgan fingerprints |
| **Solvents** | 28 | 6 solvent encoders (%) and 8 additive encoders (M) for both solvent fronts A & B |
| **Gradient profile** | 100 | Uniform gradient resampling and normalization to 100 uniform intervals |
| **Gradient duration** | 1 | Total method time in seconds |
| **Column** | 5 | 2 one hot encoders for column type (RP/HI) and 3 numerical measurements |
| **Flow rate** | 1 | Direct value (mL/min) |
| **Temperature** | 1 | Direct value (°C) |
