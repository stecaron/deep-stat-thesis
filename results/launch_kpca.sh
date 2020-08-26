#! /bin/bash

DATE=$(date '+%Y-%m-%d')

mkdir -p results/$DATE

python3 -m src.mnist.kPCA --normal-digit 1 --anomalies 5 --file "results/$DATE/kpca_scenario1.csv"
python3 -m src.mnist.kPCA --normal-digit 1 --anomalies 5 9 --file "results/$DATE/kpca_scenario2.csv"
python3 -m src.mnist.kPCA --normal-digit 1 --anomalies 0 2 3 4 5 6 7 8 9 --file "results/$DATE/kpca_scenario3.csv"

python3 -m src.mnist.kPCA --normal-digit 6 --anomalies 8 --file "results/$DATE/kpca_scenario4.csv"
python3 -m src.mnist.kPCA --normal-digit 6 --anomalies 3 8 --file "results/$DATE/kpca_scenario5.csv"
python3 -m src.mnist.kPCA --normal-digit 6 --anomalies 0 1 2 3 4 5 7 8 9 --file "results/$DATE/kpca_scenario6.csv"

python3 -m src.cars.kPCA --file "results/$DATE/kpca_cars.csv"