import os
import json
import torch
import argparse
import pandas as pd

import pickle
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.data import TabularDataset

from google.cloud import storage
#import random

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

def download_to_local(csv_uri):
    bucket_name = csv_uri
    storage_client = storage.Client()

    bucket=storage_client.get_bucket(bucket_name)
    blobs=bucket.list_blobs(prefix='rawdata/', delimiter='/') #List all objects that satisfy the filter.

    print('File download Started... Wait for the job to complete.')
    
    # Iterating through for loop one by one using API call
    for blob in blobs:
        if blob.name == 'rawdata/': continue
        print('Blobs: {}'.format(blob.name))
        
        blob.download_to_filename(blob.name.replace('rawdata/',''))
    #print(len(os.listdir()))
    
def divideGCSUri(gcsUri):
    gcsprefix = "gs://"
    assert(gcsUri.index(gcsprefix) == 0)
    bucket = gcsUri[len(gcsprefix):gcsUri.index("/", len(gcsprefix))]
    blob = gcsUri[gcsUri.index(bucket) + len(bucket)+1:]
    item = blob[blob.rindex("/")+1:]
    return bucket, blob, item

def upload_blob(bucket_name, src_file, target_blob):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(target_blob)
    blob.upload_from_filename(src_file)
    print("File {} uploaded".format(src_file))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
 
    arg_parser.add_argument(
    "--csv_uri",
    type=str,
    required=True,
    help="Preprocessed CSV training data located in GCS (gs://{path}/{filename}.csv)"
    )

    arg_parser.add_argument(
    "--checkpoint_uri",
    type=str,
    required=True,
    help="GCS URI for saving checkpoint (gs://{path}/{filename})"
    )
    
    arg_parser.add_argument(
    "--acc_uri",
    type=str,
    required=True,
    help="GCS URI for saving accuracy csv (gs://{path}/{filename}.csv)"
    )
       
    # AI Platform Job submit throws unknown arguments
    args, unknown = arg_parser.parse_known_args()
    download_to_local(args.csv_uri)
    
    #SEED = 5
    #random.seed(SEED)
    #torch.manual_seed(SEED)

    # 하이퍼파라미터
    BATCH_SIZE = 64
    lr = 0.001
    EPOCHS = 10

    # 디바이스
    print("Making model...")
    if torch.cuda.is_available():    
        DEVICE = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    TEXT = data.Field(sequential=True, batch_first=True, lower=True)
    LABEL = data.Field(sequential=False, batch_first=True)

    #trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
    trainset, testset=TabularDataset.splits(
        path=".", train="51_conan_the_barbarian.csv", test="231_Terminator.csv", format="csv",
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
    valset=testset

    TEXT.build_vocab(trainset, min_freq=5) # 단어 집합 생성
    LABEL.build_vocab(trainset)

    vocab_size = len(TEXT.vocab)
    n_classes = 8
    print('Wrod Count : {}'.format(vocab_size))
    print('Num of classes : {}'.format(n_classes))

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (trainset, valset, testset), batch_size=BATCH_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False,
            shuffle=True,
            repeat=False)
    
    model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 학습
    
    # 이전 데이터 확인
    model_bucket, model_blob, model_file = divideGCSUri(args.checkpoint_uri)
    acc_bucket, acc_blob, _ = divideGCSUri(args.acc_uri)
    best_val_loss = None
    try:
        acc_df=pd.read_csv(args.acc_uri)
        best_val_loss=acc_df["loss"].item()
        with open(model_file, 'rb') as f:
            model=pickle.load(f)
    except FileNotFoundError:
        print("No accuracy file, we will create one")
    
    # 학습
    for e in range(1, EPOCHS+1):
        train(model, optimizer, train_iter)
        val_loss, val_accuracy = evaluate(model, val_iter)

        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

        # 검증 오차가 가장 적은 최적의 모델을 저장
        if not best_val_loss or val_loss < best_val_loss:
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(model.state_dict(), './snapshot/txtclassification.pt')
            best_val_loss = val_loss
            acc_score=val_accuracy.item()/100
            
            #model
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
                upload_blob(model_bucket, model_file, model_blob)
            
            #acc
            acc_df = pd.DataFrame({"acc":acc_score, "deploy":"pending", "loss":val_loss}, index=[0])
            acc_df.to_csv(args.acc_uri)
            
            # metrics
            metrics = {
                "metrics" : [{
                "name" : "accuracy-score",
                "numberValue" : acc_score,
                "format" : "PERCENTAGE"
                }]
            }
            print("Writing matcis file: /mlpipeline-metrics.json")
            try:
                with open("/mlpipeline-metrics.json", "w") as f:
                    json.dump(metrics, f)
                    upload_blob
            except PermissionError:
                print("For local test we have no permission for /, so write current working dir:")
                with open("mlpipeline-metrics.json", "w") as f:
                    json.dump(metrics, f)

    model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
    test_loss, test_acc = evaluate(model, test_iter)
    print('Loss: %5.2f | ACC: %5.2f' % (test_loss, test_acc))
    