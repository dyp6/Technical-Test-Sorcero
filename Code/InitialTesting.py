#!/home/postd/anaconda3/bin/python

import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import list_datasets, load_dataset, ClassLabel
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def load_tokenize_dataset(path,pretrained_model='bert-base-cased'):
    hf_ds = load_dataset('csv',data_files={'train':path+'train.csv',
                                            'test':path+'test.csv',
                                            'validation':path+'dev.csv'})
    hf_ds = hf_ds.remove_columns(['claim_id','explanation','main_text','date_published',
                                'fact_checkers','sources','subjects','labels'])

    new_features = hf_ds['train'].features.copy()
    new_features['label'] = ClassLabel(names=["false","unproven","mixture","true"])
    hf_ds = hf_ds.cast(new_features)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def encode_hf(dset):
        return tokenizer(dset['claim'], truncation=True, padding='max_length')

    hf_token = hf_ds.map(encode_hf,batched=True)

    return hf_token, tokenizer

def format_dataset_splits(tokenizer,hftokenized,set_name):
    df_hf = hftokenized[set_name]
    tf_hf = df_hf.remove_columns(["claim"]).with_format("tensorflow")

    features = {x: tf_hf[x].to_tensor() for x in tokenizer.model_input_names}
    tf_tensors = tf.data.Dataset.from_tensor_slices((features, tf_hf["label"]))

    if set_name == "validation":
        tf_tensors = tf_tensors.batch(8)
    else:
        tf_tensors = tf_tensors.shuffle(len(tf_hf)).batch(8)
    
    return tf_tensors

def fine_tune_training(mod,tr_set,te_set):
    mod.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    mod.fit(tr_set, validation_data=te_set, epochs=3)
    return mod

def main():
    data_path = "/home/postd/Documents/gitRepos/Technical-Test-Sorcero/Data/RawDataCsvFormat/"
    hf_tkzd, tknzr = load_tokenize_dataset(data_path)
    
    train_hf_data = format_dataset_splits(tknzr,hf_tkzd,'train')
    test_hf_data = format_dataset_splits(tknzr,hf_tkzd,'test')
    val_hf_data = format_dataset_splits(tknzr,hf_tkzd,'validation')

    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased",num_labels=4)
    tuned_model = fine_tune_training(model,train_hf_data,test_hf_data)

    tuned_model.save_pretrained("/home/postd/Documents/gitRepos/Technical-Test-Sorcero/my_healthFacts_model")

if __name__=="__main__":
    main()
