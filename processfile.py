import os
train_lab=list()
PUC_lab="/home/zengweijia/.jupyter/cnrpark/splits/PKLot/PUC_test.txt"
UFPR04_lab="/home/zengweijia/.jupyter/cnrpark/splits/PKLot/UFPR04_test.txt"
UFPR05_lab="/home/zengweijia/.jupyter/cnrpark/splits/PKLot/UFPR04_test.txt"
target_path="/home/zengweijia/.jupyter/cnrpark/splits/PKLot/pklot_test.txt"
lab=[PUC_lab,UFPR04_lab,UFPR05_lab]
with open(target_path,"w") as fw:
    for file_path in lab:
        count=0
        with open(file_path, 'r') as f:
            count+=1
            lines=f.readlines()
            fw.writelines(lines)
    print(count)
