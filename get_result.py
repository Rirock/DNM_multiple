import os
import csv
import numpy as np

data_path = "./logs"
log_path = "./logs"
file_names = os.listdir(data_path)

# models = ["DNM_GRU_MODEL","DNM_LSTM_MODEL_8", "DNM_LSTM", "DNM_RNN_MODEL", "lstm_reg"]
# models = [ "DNM_LSTM_MODEL_M8", "DNM_RNN_MODEL", "lstm_reg", "gru_reg", "MLP", "SVM"]
#models = ["DNM_Linear_M3_M20"]
#models = ["DNM_Linear_M3_M20", "DNM_multiple_32_M20","ICFC_32"]#, "gru_reg"]
models = ["MLP_32", "MLP_64", "DNM_multiple_32_M20","ICFC_32", "KNN"]#, "gru_reg"]
models = ["DNM_multiple_32_M12", "DNM_multiple_32_M14", "DNM_multiple_32_M16", "DNM_multiple_32_M18", "DNM_multiple_32_M20"] # "DNM_multiple_32_M2", "DNM_multiple_32_M4", 

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
        #mnam = len(open(logDiri, "r").readlines())
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
    winner = np.where(r_m==np.amax(r_m))[0]
    print(winner)
    #n = np.argmax(r_m)
    for n in winner:
        jsnum[n] += 1
        print(models[n], end=", ")
    print()
    print()
    result.append(r_m)

print(jsnum)

def get_data_by_gp(path, data_name, methods):
    result_list = []
    for method in methods:
        logDiri = os.path.join(path, data_name, method+"_log.csv")
        csv_r = csv.reader(open(logDiri, "r"))
        r = []
        for i, num in enumerate(csv_r):
            r.append(num)
        r = np.array(r).astype(float)
        #rr = np.mean(r, axis=0)
        result_list.append(r)
    return(result_list)