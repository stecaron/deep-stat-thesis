#! /bin/bash

DATE="2020-10-09"
N_TRAININGS=20

mkdir -p results/$DATE/ae

for i in $(seq 1 $N_TRAININGS)
do
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 5 --folder "results/$DATE/ae/" --file "scenario1_plus" --p_train 0.01 --p_test 0.1
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 5 --folder "results/$DATE/ae/" --file "scenario1_moins" --p_train 0.1 --p_test 0.01
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 5 --folder "results/$DATE/ae/" --file "scenario1_egal" --p_train 0.05 --p_test 0.05
   
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 5 9 --folder "results/$DATE/ae/" --file "scenario2_plus" --p_train 0.01 --p_test 0.1
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 5 9 --folder "results/$DATE/ae/" --file "scenario2_moins" --p_train 0.1 --p_test 0.01
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 5 9 --folder "results/$DATE/ae/" --file "scenario2_egal" --p_train 0.05 --p_test 0.5
   
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 0 2 3 4 5 6 7 8 9 --folder "results/$DATE/ae/" --file "scenario3_plus" --p_train 0.01 --p_test 0.1
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 0 2 3 4 5 6 7 8 9 --folder "results/$DATE/ae/" --file "scenario3_moins" --p_train 0.1 --p_test 0.01
   python3 -m src.mnist.main_ae --normal-digit 1 --anomalies 0 2 3 4 5 6 7 8 9 --folder "results/$DATE/ae/" --file "scenario3_egal" --p_train 0.05 --p_test 0.05

   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 8 --folder "results/$DATE/ae/" --file "scenario4_plus" --p_train 0.01 --p_test 0.1
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 8 --folder "results/$DATE/ae/" --file "scenario4_moins" --p_train 0.1 --p_test 0.01
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 8 --folder "results/$DATE/ae/" --file "scenario4_egal" --p_train 0.05 --p_test 0.05
   
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 3 8 --folder "results/$DATE/ae/" --file "scenario5_plus" --p_train 0.01 --p_test 0.1
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 3 8 --folder "results/$DATE/ae/" --file "scenario5_moins" --p_train 0.1 --p_test 0.01
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 3 8 --folder "results/$DATE/ae/" --file "scenario5_egal" --p_train 0.05 --p_test 0.05
   
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 0 1 2 3 4 5 7 8 9 --folder "results/$DATE/ae/" --file "scenario6_plus" --p_train 0.01 --p_test 0.1
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 0 1 2 3 4 5 7 8 9 --folder "results/$DATE/ae/" --file "scenario6_moins" --p_train 0.1 --p_test 0.01
   python3 -m src.mnist.main_ae --normal-digit 6 --anomalies 0 1 2 3 4 5 7 8 9 --folder "results/$DATE/ae/" --file "scenario6_egal" --p_train 0.05 --p_test 0.05

   python3 -m src.cars.main_ae --folder "results/$DATE/ae/" --file "scenario_cars_plus" --p_train 0.01 --p_test 0.1
   python3 -m src.cars.main_ae --folder "results/$DATE/ae/" --file "scenario_cars_moins" --p_train 0.1 --p_test 0.01
   python3 -m src.cars.main_ae --folder "results/$DATE/ae/" --file "scenario_cars_egal" --p_train 0.05 --p_test 0.05
done

