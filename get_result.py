import os
import csv
import numpy as np

data_path = "./logs"
log_path = "./logs"
file_names = os.listdir(data_path)

# models = ["DNM_GRU_MODEL","DNM_LSTM_MODEL_8", "DNM_LSTM", "DNM_RNN_MODEL", "lstm_reg"]
# models = [ "DNM_LSTM_MODEL_M8", "DNM_RNN_MODEL", "lstm_reg", "gru_reg", "MLP", "SVM"]
models = ["MLP_32", "MLP_64", "DNM_Linear_M3_M20", "DNM_multiple_32_M20"]#, "gru_reg"]

nameEnd = "_log.csv"

folder_paths = []
for folder in file_names:
    folder_path = os.path.join(log_path, folder)
    if os.path.isdir(folder_path):
        folder_paths.append(folder)

result = []
jsnum = [0 for _ in range(len(models))]

for file_name in folder_paths:
    logDir = os.path.join(log_path, file_name.split(".")[0])
    print(file_name.split(".")[0])
    r_m = []
    # r_xz = []
    
    for model in models:
        logDiri = os.path.join(logDir, model+nameEnd)
        # try:
        csv_r = csv.reader(open(logDiri, "r"))
        r = []
        mnam = len(open(logDiri, "r").readlines())
        for i, num in enumerate(csv_r):
            r.append(num)
        r = np.array(r)
        r = r.astype(float)
        print("%-20s"%model, end=": \t")
        rr = np.mean(r, axis=0)
        for i in range(len(rr)):
            print("%.3f"%rr[i], end=", ")
        print()

        r_m.append(rr[2])
        # except:
        #     print("None")
    print()
    n = np.argmax(r_m)
    jsnum[n] += 1
    print(models[n])
    print()
    result.append(r_m)
    #print(r_m)
print(jsnum)
#for r in result: