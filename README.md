# FinSIR implementation on StockGraph

## Preparations

### NASDAQ and NYSE Files

Extract raw files from [Temporal-Relational-Ranking-for-Stock-Prediction](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) repository.

```
cd dataset
tar zxvf raw.tar.gz
```

Pre-process and clean raw files.

```
cd dataset
python process.py
```

## Experiments

### FinSIR (NASDAQ)

```
python train.py --market NASDAQ
```

### FinSIR (NYSE)

```
python train.py --market NYSE
```

## Summary

### NASDAQ

| Model | Test MSE | Test MRR(1) | Test IRR(1) | Parameters |
| :----: | :-------: | ----------- | ----------- | :--------: |
| FinSIR |          |             |             |            |

### NYSE

| Model | Test MSE | Test MRR(1) | Test IRR(1) | Parameters |
| :----: | :-------: | ----------- | ----------- | :--------: |
| FinSIR |          |             |             |            |
