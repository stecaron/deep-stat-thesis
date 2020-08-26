#! /bin/bash

DATE="2020-08-05"
N_TRAININGS=10

mkdir -p results/$DATE

for i in $(seq 1 $N_TRAININGS)
do
    python3 -m src.cars.main_vae_svm --file "results/$DATE/vae_svm_cars.csv"
    #python3 -m src.mnist.main_vae_svm --normal-digit 1 --anomalies 5 --file "results/$DATE/vae_svm_scenario1.csv"
    #python3 -m src.mnist.main_vae_svm --normal-digit 1 --anomalies 5 9 --file "results/$DATE/vae_svm_scenario2.csv"
    #python3 -m src.mnist.main_vae_svm --normal-digit 1 --anomalies 0 2 3 4 5 6 7 8 9 --file "results/$DATE/vae_svm_scenario3.csv"

    #python3 -m src.mnist.main_vae_svm --normal-digit 6 --anomalies 8 --file "results/$DATE/vae_svm_scenario4.csv"
    #python3 -m src.mnist.main_vae_svm --normal-digit 6 --anomalies 3 8 --file "results/$DATE/vae_svm_scenario5.csv"
    #python3 -m src.mnist.main_vae_svm --normal-digit 6 --anomalies 0 1 2 3 4 5 7 8 9 --file "results/$DATE/vae_svm_scenario6.csv"

done

