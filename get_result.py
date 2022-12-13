import os
import csv
import numpy as np

data_path = "./logs_f"
log_path = "./logs_f"
file_names = os.listdir(data_path)

# models = ["DNM_GRU_MODEL","DNM_LSTM_MODEL_8", "DNM_LSTM", "DNM_RNN_MODEL", "lstm_reg"]
# models = [ "DNM_LSTM_MODEL_M8", "DNM_RNN_MODEL", "lstm_reg", "gru_reg", "MLP", "SVM"]
models = ["DNM_LSTM_MODEL_M2", "DNM_LSTM_MODEL_M4", "DNM_LSTM_MODEL_M5", "DNM_LSTM_MODEL_M6", "DNM_LSTM_MODEL_M8", "DNM_LSTM_MODEL_M10", "DNM_LSTM_MODEL_M15", "DNM_LSTM_MODEL_M20"]#, "gru_reg"]

nameEnd = "_pred.csv"

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
        r = [[] for _ in range(4)]
        mnam = len(open(logDiri, "r").readlines())
        for i, num in enumerate(csv_r):
            nnnn = i % 4
            r[nnnn].append(float(num[0]))

        for i, rr in enumerate(r[-1]):
            if float(rr) < 0:
                for j in range(len(r)):
                    del r[j][i]
        n = np.argmin(r[0])
        for i in range(len(r)):
            #print(np.mean(r[i]), end=", ")
            print(r[i][n], end=", ")
        print()
        r_m.append(r[0][n])

        # r_m.append(np.mean(r[0]))
        # except:
        #     print("None")
    n = np.argmin(r_m)
    jsnum[n] += 1
    print(models[n])
    print()
    result.append(r_m)
    #print(r_m)
print(jsnum)
#for r in result: