import pandas as pd
from glob import glob
import re

def extractPreds():
    model_dirs = glob("./output_files/*")
    all_models = []
    for d in model_dirs:
        all_models.append(glob(d+"/*"))
    model_paths = [x for y in all_models for x in y]
    model_names = []
    for mpath in model_paths:
        mn = re.findall("output_files/.*/",mpath)[0][13:-1]
        if mn not in model_names:
            model_names.append(mn)
        
    te = pd.read_csv("../Data/RawDataCsvFormat/claimLabel_test.csv")
    
    for mname in model_names:
        files = [x for x in model_paths if mname in x]
        for fname in files:
            label = re.findall("_\d{1,3}_\d{1,3}_\d{1,3}",fname)[0]
            f = open(fname,'r')
            preds = []
            for line in f:
                preds.append(line)
    
            predsDF = pd.DataFrame({"index":[x.split('\t')[0] for x in preds[1:]],
                                "prediction":[x.split('\t')[1].replace('\n','')\
                                              for x in preds[1:]]})
            for i in predsDF.index:
                te.loc[i,mname+label] = predsDF.loc[i,"prediction"] 
    
    return te, model_names

def predResults(model_name,testDF):
    colnames = re.findall(" "+model_name+"_\d{1,3}_\d{1,3}_\d{1,3}",
                          " ".join([x for x in testDF.columns]))
    colnames = [x.strip() for x in colnames]
    
    tmp = {"model":[],"total":[],"true":[],
          "mixture":[],"unproven":[],"false":[]}
    
    for mod in colnames:
        tmp["model"].append(mod)
        
        correct = []
        for i in range(len(testDF)):
            if testDF.loc[i,"label"] == testDF.loc[i,mod]:
                correct.append(i)
        tmp["total"].append(len(correct)/len(testDF))
        
        for lab in list(tmp.keys())[2:]:
            observed = testDF.loc[testDF.label==lab,:]
            g = 0
            for i in observed.index:
                if observed.loc[i,"label"] == observed.loc[i,mod]:
                    g+=1
            tmp[lab].append(g/len(observed))
            
        tmpDF = pd.DataFrame(tmp)
        
    return tmpDF


def main():
    test_preds, mod_names = extractPreds()
    
    results = pd.DataFrame({"model":[],"total":[],"true":[],
                               "mixture":[],"unproven":[],"false":[]})
    
    for mname in mod_names:
        results = pd.concat([results,predResults(mname,test_preds)])
    
    results.to_csv("./output_files/testing_accuracy.csv",index=False)

if __name__ == "__main__":
    main()
