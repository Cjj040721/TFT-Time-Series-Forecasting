# Third-Party Notices

The root MIT License applies only to original project code and documentation. The thesis is separately licensed under CC BY 4.0. Third-party components and external datasets are not relicensed by this project.

## Informer2020

The directory `third_party/informer2020/` contains the upstream implementation from [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020), retained for the Informer baseline.

The upstream [Apache License 2.0](third_party/informer2020/LICENSE) and README are preserved. Training data, checkpoints, and generated results from the upstream project are not included.

Reference: Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. [arXiv:2012.07436](https://arxiv.org/abs/2012.07436).

## Temporal Fusion Transformer

The TFT implementation in `src/model/tft.py` was written for this thesis project based on the model described in:

Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, International Journal of Forecasting, 2021. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363).

## Datasets

ETT is available from [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset). Weather and ECL dataset references are preserved through the Informer2020 documentation. Raw datasets are not redistributed by this repository.
