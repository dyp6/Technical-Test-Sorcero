import pandas as pd

path = "C://Users/postd/Documents/Personal_Stuff/Technical-Test-Sorcero-main/"

test = pd.read_csv(path+"Data/RawDataCsvFormat/claimLabel_test.csv")

def extractPreds(model_name):
    preds = []
    f =  open(path+"Code/output_files/"+model_name+"/test_results.txt",'r')

    preds = []
    for line in f:
        preds.append(line)

    predsDF = pd.DataFrame({"index":[x.split('\t')[0] for x in preds[1:]],
                            "prediction":[x.split('\t')[1].replace('\n','')\
                                          for x in preds[1:]]})
    return predsDF


bert_preds = extractPreds('bert')
distilbert_preds = extractPreds('distilbert')

def predResults(model_name,testDF,predDF):
    correct = []
    wrong = []
    for i in range(len(test)):
        if testDF.loc[i,"label"] == predDF.loc[i,'prediction']:
            correct.append(i)
        else:
            wrong.append(i)
    

    print(str((len(correct)/len(test)) * 100) + "% claims correctly predicted by "
          +model_name+".")
    print(str((len(wrong)/len(test)) * 100) + "% claims incorrectly predicted by "
          +model_name+".")



predResults("bert",test,bert_preds)

print("\n")

predResults('distilbert',test,distilbert_preds)
