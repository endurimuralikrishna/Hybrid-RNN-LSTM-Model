"""Reconstructed text-only experiment; not the original manuscript implementation."""
import argparse,json,re,html,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def clean(s):
    s=html.unescape(str(s)).lower()
    s=re.sub(r'https?://\S+|www\.\S+|(?:u/|@)\w+',' ',s)
    return ' '.join(re.sub(r'[^a-z0-9\s]',' ',s).split())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--text-column',required=True); p.add_argument('--label-column',required=True); p.add_argument('--positive-label',required=True); p.add_argument('--negative-label',required=True); p.add_argument('--word2vec'); p.add_argument('--models',nargs='+',choices=['lr','svm','rnn_lstm','lstm','bilstm','cnn','bert'],default=['rnn_lstm']); p.add_argument('--output',default='outputs'); a=p.parse_args()
    c=json.loads(Path('config.json').read_text()); seed=c['seed']; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.csv,dtype=str).dropna(subset=[a.text_column,a.label_column]); d=d[[a.text_column,a.label_column]].copy(); d.columns=['text','label']; d['text']=d.text.map(clean); d=d[d.text!='']; mapping={a.negative_label:0,a.positive_label:1}
    if len(mapping)!=2 or not set(d.label).issubset(mapping): raise ValueError('Check label mapping against dataset documentation.')
    d['label']=d.label.map(mapping); n=len(d); conflict=d.groupby('text').label.transform('nunique')>1; conflicts=int(conflict.sum()); d=d[~conflict].drop_duplicates('text').reset_index(drop=True)
    train,test=train_test_split(d,test_size=c['test_fraction'],stratify=d.label,random_state=seed); train,val=train_test_split(train,test_size=c['validation_fraction_of_training'],stratify=train.label,random_state=seed)
    audit={'clean_nonempty_rows':n,'conflicting_rows_removed':conflicts,'unique_rows':len(d),'train':len(train),'validation':len(val),'test':len(test),'input_sha256':hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(),'config':c,'label_mapping':mapping}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)); print(audit)
    results=[]
    for name in a.models:
        if name in ['lr','svm']:
            model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=20000),LogisticRegression(max_iter=1000,random_state=seed) if name=='lr' else LinearSVC(random_state=seed)); model.fit(train.text,train.label); pred=model.predict(test.text)
        elif name=='bert':
            import torch
            from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments
            from torch.utils.data import Dataset
            tok=AutoTokenizer.from_pretrained('bert-base-uncased')
            class Posts(Dataset):
                def __init__(self,frame): self.enc=tok(frame.text.tolist(),truncation=True,padding='max_length',max_length=c['sequence_length']); self.y=frame.label.tolist()
                def __len__(self): return len(self.y)
                def __getitem__(self,i): return {**{k:torch.tensor(v[i]) for k,v in self.enc.items()},'labels':torch.tensor(self.y[i])}
            torch.manual_seed(seed); model=AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2); trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(out/'bert_checkpoints'),num_train_epochs=c['epochs'],per_device_train_batch_size=8,per_device_eval_batch_size=8,seed=seed,report_to=[],save_strategy='no'),train_dataset=Posts(train)); trainer.train(); pred=trainer.predict(Posts(test)).predictions.argmax(axis=1)
        else:
            if not a.word2vec: p.error('--word2vec required for recurrent/CNN models')
            import tensorflow as tf
            from gensim.models import KeyedVectors
            tf.keras.utils.set_random_seed(seed)
            tok=tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>'); tok.fit_on_texts(train.text.tolist())
            def encode(frame): return tf.keras.utils.pad_sequences(tok.texts_to_sequences(frame.text.tolist()),maxlen=c['sequence_length'],padding='post',truncating='post')
            vectors=KeyedVectors.load_word2vec_format(a.word2vec,binary=True); dim=c['embedding_dimension']; assert vectors.vector_size==dim
            weights=np.random.default_rng(seed).normal(0,.05,(len(tok.word_index)+1,dim)).astype('float32'); weights[0]=0
            for word,i in tok.word_index.items():
                if word in vectors: weights[i]=vectors[word]
            del vectors
            L=tf.keras.layers; inp=L.Input((c['sequence_length'],),dtype='int32'); x=L.Embedding(len(weights),dim,weights=[weights],trainable=False,mask_zero=name!='cnn')(inp)
            if name=='rnn_lstm': x=L.SimpleRNN(c['rnn_units'],return_sequences=True)(x); x=L.LSTM(c['lstm_units'])(x)
            elif name=='lstm': x=L.LSTM(c['lstm_units'])(x)
            elif name=='bilstm': x=L.Bidirectional(L.LSTM(c['lstm_units']))(x)
            else: x=L.Conv1D(64,3,activation='relu')(x); x=L.GlobalMaxPooling1D()(x)
            x=L.Dropout(c['dropout'])(x); model=tf.keras.Model(inp,L.Dense(1,activation='sigmoid')(x)); model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']); h=model.fit(encode(train),train.label,validation_data=(encode(val),val.label),epochs=c['epochs'],batch_size=c['batch_size'],verbose=2); pred=(model.predict(encode(test)).ravel()>=.5).astype(int)
            pd.DataFrame(h.history).to_csv(out/f'{name}_history.csv',index=False)
            for metric in ['loss','accuracy']:
                plt.figure(); plt.plot(range(1,c['epochs']+1),h.history[metric],label='Training'); plt.plot(range(1,c['epochs']+1),h.history['val_'+metric],label='Validation'); plt.xlabel('Epoch'); plt.ylabel(metric); plt.legend(); plt.tight_layout(); plt.savefig(out/f'{name}_{metric}.png',dpi=300); plt.close()
            tf.keras.backend.clear_session()
        report=classification_report(test.label,pred,labels=[0,1],output_dict=True,zero_division=0); cm=confusion_matrix(test.label,pred,labels=[0,1]); (out/f'{name}_metrics.json').write_text(json.dumps({'report':report,'confusion_matrix':cm.tolist()},indent=2)); ConfusionMatrixDisplay(cm,display_labels=['Non-depressive','Depressive']).plot(); plt.tight_layout(); plt.savefig(out/f'{name}_confusion.png',dpi=300); plt.close(); results.append({'model':name,'accuracy':accuracy_score(test.label,pred),**{k:report['macro avg'][k] for k in ['precision','recall','f1-score']}})
    pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
if __name__=='__main__': main()
