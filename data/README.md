# Datasets

Raw datasets are not tracked in Git. Download them from their original sources and place them at the paths below.

| Dataset | Expected path | Source |
| --- | --- | --- |
| ETTh1 | `data/raw/ETT/ETTh1.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTh2 | `data/raw/ETT/ETTh2.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm1 | `data/raw/ETT/ETTm1.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | `data/raw/Weather/WTH.csv` | See the [Informer data instructions](https://github.com/zhouhaoyi/Informer2020#data) |
| ECL | `data/raw/ECL/ECL.csv` | See the [Informer data instructions](https://github.com/zhouhaoyi/Informer2020#data) |

Each CSV must contain a `date` column. ETT files use the columns `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, and `OT`; `OT` is the forecasting target.

The local candidate directory may contain an ignored ETTh1 file used for verification. That local copy is not part of the proposed Git history.
