#!/bin/bash

log=tva_fold1.txt
# create log file or overwrite if already present
printf "Log File - " > $log
# append date to log file
date >> $log

for i in 1 2 3 4 5 6 7 8 9 10

do
  echo $i
  python main_tva_1.py --seed $i --data_path '/processed_Yoon/IEMOCAP/seven_category_120/folds/fold01' >> $log

done

log=tva_fold2.txt
# create log file or overwrite if already present
printf "Log File - " > $log
# append date to log file
date >> $log

for i in 1 2 3 4 5 6 7 8 9 10

do
  echo $i
  python main_tva_1.py --seed $i --data_path '/processed_Yoon/IEMOCAP/seven_category_120/folds/fold02' >> $log

done

log=tva_fold3.txt
# create log file or overwrite if already present
printf "Log File - " > $log
# append date to log file
date >> $log

for i in 1 2 3 4 5 6 7 8 9 10

do
  echo $i
  python main_tva_1.py --seed $i --data_path '/processed_Yoon/IEMOCAP/seven_category_120/folds/fold03' >> $log

done

log=tva_fold4.txt
# create log file or overwrite if already present
printf "Log File - " > $log
# append date to log file
date >> $log

for i in 1 2 3 4 5 6 7 8 9 10

do
  echo $i
  python main_tva_1.py --seed $i --data_path '/processed_Yoon/IEMOCAP/seven_category_120/folds/fold04' >> $log

done

log=tva_fold5.txt
# create log file or overwrite if already present
printf "Log File - " > $log
# append date to log file
date >> $log

for i in 1 2 3 4 5 6 7 8 9 10

do
  echo $i
  python main_tva_1.py --seed $i --data_path '/processed_Yoon/IEMOCAP/seven_category_120/folds/fold05' >> $log

done
