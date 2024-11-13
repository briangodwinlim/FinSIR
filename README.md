# FinSIR implementation on StockGraph

## Preparations

### NYSE and NASDAQ Files

Extract raw files from the [Temporal-Relational-Ranking-for-Stock-Prediction](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) repository.

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

### NYSE

```
python train.py --nworkers 2 --use-amp --market NYSE --sequence-length 8 --add-self-loop --relational-graph wiki --model SimpleFinSIR --nhidden 8 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 0.1

python train.py --nworkers 2 --use-amp --market NYSE --sequence-length 8 --add-self-loop --relational-graph wiki --model FinSIR --nhidden 16 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 0.1

python train.py --nworkers 2 --use-amp --market NYSE --sequence-length 16 --add-self-loop --relational-graph industry --model SimpleFinSIR --nhidden 64 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 0.1

python train.py --nworkers 2 --use-amp --market NYSE --sequence-length 8 --add-self-loop --relational-graph industry --model FinSIR --nhidden 8 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 0.1
```

### NASDAQ

```
python train.py --nworkers 2 --use-amp --market NASDAQ --sequence-length 2 --add-self-loop --relational-graph wiki --model SimpleFinSIR --nhidden 8 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 0.1

python train.py --nworkers 2 --use-amp --market NASDAQ --sequence-length 2 --add-self-loop --relational-graph wiki --model FinSIR --nhidden 32 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 1.0

python train.py --nworkers 2 --use-amp --market NASDAQ --sequence-length 2 --add-self-loop --relational-graph industry --model SimpleFinSIR --nhidden 16 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 0.1

python train.py --nworkers 2 --use-amp --market NASDAQ --sequence-length 2 --add-self-loop --relational-graph industry --model FinSIR --nhidden 16 --recurrent-layers 1 --recurrent-dropout 0 --relational-agg sym --relational-dropout 0 --readout-layers 1 --readout-dropout 0 --epochs 50 --lr 1e-3 --wd 1e-5 --factor 0.5 --patience 50 --alpha 0.1
```

## Summary

### NYSE

| Model                      |      IRR 1      |      IRR 5      |      MRR 1      |      MRR 5      |           MSE           |
| :------------------------- | :--------------: | :--------------: | :--------------: | :--------------: | :----------------------: |
| RankLSTM                   | 0.0140 ± 0.2322 | 0.0605 ± 0.1096 | 0.0260 ± 0.0080 | 0.0168 ± 0.0040 | 1.2632e-02 ± 2.7230e-03 |
| RSR-I (Wiki)               | 0.6148 ± 0.6462 | 0.4465 ± 0.2170 | 0.0265 ± 0.0066 | 0.0234 ± 0.0021 | 2.0209e-02 ± 1.4140e-03 |
| RSR-E (Wiki)               | 0.9491 ± 0.4731 | 0.4075 ± 0.2395 | 0.0339 ± 0.0058 | 0.0226 ± 0.0015 | 1.9099e-02 ± 1.0490e-03 |
| SimpleFinSIR (Wiki)        | 1.3457 ± 0.4256 | 0.5673 ± 0.1267 | 0.0369 ± 0.0068 | 0.0244 ± 0.0022 | 1.9771e-02 ± 1.3940e-03 |
| FinSIR (Wiki)              | 1.6034 ± 0.3451 | 0.4337 ± 0.1251 | 0.0298 ± 0.0043 | 0.0200 ± 0.0027 | 1.8302e-02 ± 2.1780e-03 |
| RSR-I (industry)           | 1.1937 ± 0.4384 | 0.4734 ± 0.1135 | 0.0348 ± 0.0062 | 0.0229 ± 0.0014 | 1.9566e-02 ± 1.1450e-03 |
| RSR-E (industry)           | 1.2093 ± 0.3199 | 0.4335 ± 0.1344 | 0.0362 ± 0.0059 | 0.0235 ± 0.0020 | 1.9915e-02 ± 1.2390e-03 |
| SimpleFinSIR (industry)    | 1.2739 ± 0.3823 | 0.4796 ± 0.1325 | 0.0343 ± 0.0045 | 0.0221 ± 0.0019 | 1.8905e-02 ± 1.1480e-03 |
| FinSIR (industry)          | 1.4761 ± 0.4458 | 0.5338 ± 0.1504 | 0.0353 ± 0.0059 | 0.0246 ± 0.0022 | 2.0035e-02 ± 1.2010e-03 |

### NASDAQ

| Model                      |      IRR 1      |      IRR 5      |      MRR 1      |      MRR 5      |           MSE           |
| :------------------------- | :--------------: | :--------------: | :--------------: | :--------------: | :----------------------: |
| RankLSTM                   | 0.2882 ± 0.2457 | 0.1485 ± 0.1269 | 0.0340 ± 0.0068 | 0.0201 ± 0.0042 | 1.5136e-02 ± 2.7060e-03 |
| RSR-I (Wiki)               | 0.2476 ± 0.3306 | 0.0630 ± 0.1417 | 0.0308 ± 0.0070 | 0.0175 ± 0.0038 | 1.5520e-02 ± 2.6230e-03 |
| RSR-E (Wiki)               | 0.2085 ± 0.4255 | 0.2044 ± 0.0776 | 0.0299 ± 0.0079 | 0.0188 ± 0.0042 | 1.5084e-02 ± 2.7480e-03 |
| SimpleFinSIR (Wiki)        | 1.1161 ± 0.3207 | 0.4460 ± 0.1685 | 0.0472 ± 0.0039 | 0.0262 ± 0.0042 | 2.1342e-02 ± 2.7510e-03 |
| FinSIR (Wiki)              | 0.7838 ± 0.5050 | 0.3051 ± 0.1583 | 0.0408 ± 0.0050 | 0.0241 ± 0.0034 | 1.9601e-02 ± 3.5320e-03 |
| RSR-I (industry)           | 0.5934 ± 0.3493 | 0.2983 ± 0.1274 | 0.0309 ± 0.0033 | 0.0220 ± 0.0020 | 1.8577e-02 ± 1.4800e-03 |
| RSR-E (industry)           | 1.1114 ± 0.2703 | 0.5670 ± 0.0880 | 0.0462 ± 0.0052 | 0.0273 ± 0.0039 | 2.1336e-02 ± 2.4930e-03 |
| SimpleFinSIR (industry)    | 0.9334 ± 0.3608 | 0.3106 ± 0.1055 | 0.0429 ± 0.0058 | 0.0242 ± 0.0046 | 1.9271e-02 ± 3.2150e-03 |
| FinSIR (industry)          | 1.2307 ± 0.3761 | 0.6747 ± 0.1374 | 0.0487 ± 0.0050 | 0.0310 ± 0.0021 | 2.4034e-02 ± 1.7680e-03 |
