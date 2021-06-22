import pandas as pd

def extractPreds(model_name):
    preds = []
    f =  open("output_files/"+model_name+"/test_results.txt",'r')

    preds = []
    for line in f:
        preds.append(line)

    predsDF = pd.DataFrame({"index":[x.split('\t')[0] for x in preds[1:]],
                            "prediction":[x.split('\t')[1].replace('\n','')\
                                          for x in preds[1:]]})
    return predsDF
    
def predResults(model_name,testDF):
    correct = []
    colname = model_name+'_preds'
    for i in range(len(testDF)):
        if testDF.loc[i,"label"] == testDF.loc[i,colname]:
            correct.append(i)
    
    labs = ["true",'mixture','unproven','false']
    labSpecAcc = []
    for lab in labs:
        observed = testDF.loc[testDF.label==lab,:]
        g = 0
        for i in observed.index:
            if observed.loc[i,"label"] == observed.loc[i,colname]:
                g+=1
        labSpecAcc.append([lab,g/len(observed)])
        
    
    print(str((len(correct)/len(testDF)) * 100) +
          "% of all claims correctly predicted by "
          +model_name+".")
    
    for i in range(len(labSpecAcc)):
        print(str(labSpecAcc[i][1]*100) + "% of "+labSpecAcc[i][0]+
              " claims correctly predicted by "+model_name+".")

def main():
    # Working directory needs to be set to the Code folder of the repository
    test = pd.read_csv("../Data/RawDataCsvFormat/claimLabel_test.csv")
    
    bert_preds = extractPreds('bert')
    distilbert_preds = extractPreds('distilbert')
    ElDisc_preds = extractPreds('electra_discriminator')
    
    for i in test.index:
        test.loc[:,"bert_preds"] = bert_preds.loc[:,"prediction"]
        test.loc[:,"distilbert_preds"] = distilbert_preds.loc[:,"prediction"]
        test.loc[:,"electra_discriminator_preds"] = ElDisc_preds.loc[:,"prediction"]
    
    print("BERT (Cased)")    
    predResults("bert",test)
    
    print("\nDistilBERT (Cased)")
    predResults('distilbert',test)

    print("\nElectra (Discriminator)\n")
    predResults('electra_discriminator',test)
    
if __name__ == "__main__":
    main()
