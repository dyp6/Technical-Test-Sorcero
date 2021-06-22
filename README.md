# Technical Test for Sorcero Position
This repository contains all the files needed to execute and understand my submission to complete the AI infrastructure and ML engineer technical test provided in the application process for Sorcero

# Technical Test Assignment
**Task:** Transfer learning for text classification

**Description:** Using one of the pretrained transformers available in huggingface, build a model to learn how to classify healthcare claims, weeding the facts from the myths). Demonstrate the performance of the final model.

**Useful links:**
* Huggingface transformers: https://huggingface.co/models
* Huggingface dataset HEALTH_FACT: https://huggingface.co/datasets/health_fact

# Repository Structure and Code Execution
### Repository
I broke the repository into a Code and a Data directory because it is and easy way to organize files at first, though I will likely reorganize it. The Code directory has all of the scripts that I used and made to run the text-classification models and extract the resulting accuracies. The Data directory is just two folders, one with the Raw .tsv files and the other has cleaned .csv files that were used as input for the models.

### Code Execution
The fine-tuned model files that were saved after training on the dataset were too large to be saved in a GitHub repository, so there is no way to just run the prediction on the test file. The overall instructions for using the run_text_classification.py script, that executes the models, is about the same as at this link https://github.com/huggingface/transformers/tree/master/examples/tensorflow/text-classification. However, the arguments for the input files are "--train_file claimLabel_train.csv", "--validation_file claimLabel_dev.csv", "--test_file claimLabel_test.csv", and "--output_dir output_files/<model_name>" (this directory needed to be outside of the Git Repository, so the models weren't saved inside it). The other argument, not discussed at that link, that I put in the "python run_text_classification.py " command was "--num_training_epochs 4"

The ManualTextClassification.py file is not complete yet, but that is the script that I plan to use to run the XLNet and T5 models later on and use the same style as the run_text_classification.py example script I used above.
# Results and Discussion
### Overview
Based on the reading I did about transformer models and text-classification, I found six transformers on huggingface.co/models that I wanted to test on this task. Those models were
* BERT (Cased)
* DistilBERT (Cased)
* Electra (Discriminator)
* Electra (Generator)
* Text-to-Text Transfer Transformer (T5)
* XLNet (Google Brain)
These are the models that are described as the state-of-the-art in text-classification as well as most other benchmark Natural Language Processing tasks.
### Data Cleaning
Before I started modeling, I looked through the data and noticed some minor issues that I felt like I should address. I noticed that in the .tsv files, for between 20 and 30 observations there was an issue with a character that caused the data to shift in the columns and have the incorrect values. There wasn't a clear programmatic way to solve this, and it wasn't too many observations, so I converted the train.tsv, dev.tsv, and test.tsv to .csv format and corrected the observations with errors in Excel. There were still about 9 unlabeled observations between the datasets, but this was much better than the 20 to 30 observations that would have been unlabeled otherwise.
### Methodology
Following the example script at https://github.com/huggingface/transformers/tree/master/examples/tensorflow/text-classification (called run_text_classification.py in the repository), I was able to run BERT, DistilBERT, and Electra models with much modification using the AutoTokenizer and AutoModel functions to load the pre-trained models. However, the XLNet and Text-to-Text Transfer Transformer were not able to be easily plugged into this example script, so I began writing a script that would allow the testing of these two models on this task. I ran each of the models with a batch size of 8, for 4 epochs, using tensorflow's AdamOptimizer and a learning rate of 2e-5 on the training and validation/development data sets to fine-tune the models for the task. Then, used those models to make predictions on the test data set, and those are the accuracies that I will present below.

I was only able to get through this process for BERT (Cased), DistilBERT (Cased), and Electra (Discriminator), so the table below presents a look at their prediction accuracies. I present the overall prediction accuracy, along with the accuracy specific to each label category for the claims. For example, "True Claims Predicted Correctly" represents the proportion of times that the model predicted "True" when the claim was actually "True". I believe these category accuracies offer a little more insight into the actual performance of the model and what next steps to take.
### Results
In the table below, the bolded values represent the values that are highest accuracy for that particular category.

| *Model Name* | *Total Claims Predicted Correctly* | *True Claims Predicted Correctly* | *Mixture Claims Predicted Correctly* | *Unproven Claims Predicted Correctly* | *False Claims Predicted Correctly* |
| --- | --- | --- | --- | --- | --- |
| BERT (Cased) | 0.5814 | **0.86** | 0 | 0 | 0.5913 |
| DistilBERT (Cased) | 0.5967 | 0.7017 | **0.4627** | **0.2** | 0.5527 |
| Electra (Discriminator) | **0.6381** | 0.7733 | 0.01 | 0.1333 | **0.8123** |
### Discussion
It is really interesting how each of these models performed comparatively. They all have fairly similar total accuracy, with Electra (Discriminator) performing slightly better than either of the other models. However, when you look into the category specific accuracies, there is a little more to notice. Most notable is the BERT model did not predict any "mixed" or "unproven" correctly (or at all), and Electra performed very poorly for these two categories as well, with 1% and 13% correct respectively. DistilBERT did not perform well on those two categories, but much higher than either of the other two models. Those two categories, logically, would be the most difficult, but it is interesting that these models showed such drastically different results.
### Future Steps
I plan to keep working on this in my free time because I have really enjoyed it, so I am going to note a few of the next steps I am planning to take. First, I am going to get each of the models executed initially and have a full comparison of their accuracies, similar to above. I did not get much time to tune hyperparameters and other modeling options, like batch-size, learning rates, learning decay, number of epochs, etc. I plan to take the two or three best models and see how adjusting there parameters effects their performance.

The split between performance on True and False categories and Mixture and Unproven categories is interesting to me as well. I want to test the models on the dataset with a few different category specifications as well. Specifically, I want to do one with only True and Untrue, where {False, Mixture, and Unproven} are all classified as Untrue (meaing not 100% true), then I want to do a three category label, with {True, False, and Unsure} where Mixture and Unproven are both in the Unsure (or Undecided) category. This will help me see what types of claims are the most difficult for these models to distinguish between and look if there are patterns within them that may help me better fine-tune the models.

I will continue to post those updates to this repository. There also are not many, if any at all, find-tuned models on huggingface.co that aim to distinguish between fact, truth, a mixture, or unproven, and I believe that would be a valuable addition to the models if I am able to develop a model that performs consistently well on this data.

# Dataset Reference
@inproceedings{kotonya-toni-2020-explainable,  
&ensp;&emsp;&emsp;title = "Explainable Automated Fact-Checking for Public Health Claims",\
&ensp;&emsp;&emsp;author = "Kotonya, Neema  and\
&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;Toni, Francesca",\
&ensp;&emsp;&emsp;booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",\
&ensp;&emsp;&emsp;month = nov,\
&ensp;&emsp;&emsp;year = "2020",\
&ensp;&emsp;&emsp;address = "Online",\
&ensp;&emsp;&emsp;publisher = "Association for Computational Linguistics",\
&ensp;&emsp;&emsp;url = "https://www.aclweb.org/anthology/2020.emnlp-main.623", \
&ensp;&emsp;&emsp;pages = "7740--7754",\
}
