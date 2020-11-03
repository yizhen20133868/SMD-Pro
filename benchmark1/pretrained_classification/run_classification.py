import random
import math
from tqdm import tqdm
from copy import deepcopy
from transformers import (
    BertConfig,BertTokenizer,
    RobertaConfig,RobertaTokenizer,
    AlbertConfig,AlbertTokenizer,
    ElectraConfig,ElectraTokenizer
)
from benchmark1.pretrained_classification.model import *
from argparse import ArgumentParser
import json
import wandb
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import TensorDataset
from transformers.optimization import AdamW,get_linear_schedule_with_warmup
from benchmark1.pretrained_classification.util import get_max_row, read_examples, convert_examples_to_features
from benchmark1.pretrained_classification.metrics import *
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMultiLabelClassification, BertTokenizer, "bert-base-cased"),
    "roberta": (RobertaConfig, RoBERTaForMultiLabelClassification, RobertaTokenizer, "roberta-base"),
    "albert": (AlbertConfig, ALbertForMultiLabelClassification, AlbertTokenizer, "albert-base-v2"),
    "electra": (ElectraConfig, ElectraForMultiLabelClassification, ElectraTokenizer, "google/electra-base-discriminator"),
}

class Classifier():

    def __init__(self,args):
        self.args = args
        # settings
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # model
        assert self.args.model_name in MODEL_CLASSES
        config_class, model_class, tokenizer_class, plm_name = MODEL_CLASSES[self.args.model_name]
        tokenizer = tokenizer_class.from_pretrained(plm_name)
        bert_config = config_class.from_pretrained(plm_name)
        num_labels = get_max_row(self.args.data_dir) + 1
        bert_config.num_labels = num_labels
        self.model = model_class.from_pretrained(plm_name, config=bert_config)
        # load data
        train_examples = read_examples(self.args.data_dir,split="train")
        dev_examples = read_examples(self.args.data_dir,split="dev")
        test_examples = read_examples(self.args.data_dir,split="test")
        tags = {
            "cls":tokenizer.cls_token,
            "sep":tokenizer.sep_token,
            "key": tokenizer.sep_token,
            "value": tokenizer.sep_token,
            "pad":tokenizer.pad_token
        }
        train_features = convert_examples_to_features(train_examples, tokenizer,max_seq_len=self.args.max_seq_length,tags=tags)
        dev_features = convert_examples_to_features(dev_examples, tokenizer,max_seq_len=self.args.max_seq_length,tags=tags)
        test_features = convert_examples_to_features(test_examples, tokenizer,max_seq_len=self.args.max_seq_length,tags=tags)
        self.train_dataloader = self.dataloader(train_features,self.args.batch_size)
        self.dev_dataloader = self.dataloader(dev_features, self.args.batch_size)
        self.test_dataloader = self.dataloader(test_features, self.args.batch_size)
        self.text_line = len(train_features)

    def dataloader(self, features, batch_size, is_train = True):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label_ids = []
        for feature in features:
            input_ids.append(feature.input_ids)
            attention_mask.append(feature.attention_mask)
            token_type_ids.append(feature.token_type_ids)
            label_ids.append(feature.label_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        params = [input_ids, attention_mask, token_type_ids, label_ids]
        dataset = TensorDataset(*params)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

    def evaluate(self,dev_dataloader):
        if int(self.args.threshold) != -1:
            acc_metric = ExactThreshAccuracy(thresh=self.args.threshold)
        else:
            acc_metric = AutoThreshAccuracy()
        all_logits = []
        all_label_ids = []
        with open("{}_results.txt".format(self.args.model_name), "w") as f:
            f.write("++start++\n")
            f.close()
        with torch.no_grad():
            for batch in tqdm(dev_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, label_ids = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
                logits = outputs[1]
                logits = logits.cpu()
                label_ids = label_ids.cpu()
                all_logits.append(logits)
                all_label_ids.append(label_ids)
                acc_metric(logits, label_ids)
            accuracy, best_thresh = acc_metric.value()
        with open("{}_results.txt".format(self.args.model_name), "a") as f:
            f.write("accuracy:{}, best_thresh:{}\n".format(accuracy, best_thresh))
            for (logits, label_ids) in zip(all_logits, all_label_ids):
                logits = torch.sigmoid(logits)
                for batch_n in range(len(label_ids)):
                    pred = []
                    real = []
                    assert len(logits[batch_n]) == len(label_ids[batch_n])
                    for j, (p, r) in enumerate(zip(logits[batch_n], label_ids[batch_n])):
                        if p > best_thresh:
                            pred.append(j)
                        if r == 1:
                            real.append(j)
                    f.write("pred:{}, real:{}\n".format(pred,real))

        return accuracy

    def train(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        num_train_optimization_steps = math.ceil(self.text_line / self.args.batch_size)  * self.args.epochs
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        schedule = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=self.args.warmup_steps,num_training_steps=num_train_optimization_steps)

        checkpoint_dir = self.args.saved_model_dir +  self.args.saved_model_name
        if self.args.load_weights or self.args.just_eval:
            self.model.load_state_dict(torch.load(checkpoint_dir) if self.args.restore_from == None else (self.args.saved_model_dir + self.args.restore_from))

        if self.args.just_eval:
            print("===== hyper parameters =====")
            print("learning rate: {}, batch size: {}, model: {}".format(self.args.lr, self.args.batch_size,self.args.model_name))
            acc= self.evaluate(self.dev_dataloader)
            print("===== evaluate set result =====")
            print("acc: %.2f" % (acc * 100))
            acc= self.evaluate(self.test_dataloader)
            print("===== test set result =====")
            print("acc: %.2f" % (acc * 100))
            return

        patience = 0
        best_acc = 0.0
        best_model_state_dict = None

        if self.args.use_wandb:
            # TODO login and init your own wandb project
            wandb.init(project="kvret")
        # start training
        for i in range(self.args.epochs):
            iterator = tqdm(self.train_dataloader)
            for step, batch in enumerate(iterator):
                self.model.train()
                optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, token_type_ids, label_ids = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids, labels=label_ids)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                schedule.step()
                if self.args.use_wandb:
                    wandb.log({"loss":loss.cpu()})
                iterator.set_description("Epoch {}, Loss: {}".format(i+1, loss.cpu()))
            # start evaluating
            acc = self.evaluate(self.dev_dataloader)
            logs = {"acc": acc}
            if self.args.use_wandb:
                wandb.log(logs)
            if acc > best_acc:
                best_acc = max(acc, best_acc)
                patience = 0
                best_model_state_dict = self.model.state_dict()
            else:
                patience += 1
                if patience >= self.args.max_patience:
                    print("run out of patience.....")
                    torch.save(best_model_state_dict, checkpoint_dir)
                    return
        torch.save(best_model_state_dict, checkpoint_dir)


if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return False
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=11111)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--max_patience', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=2000)

    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--threshold', type=float, default=-1)
    parser.add_argument('--restore_from', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="bert")
    parser.add_argument('--load_weights', type=str2bool, default=False)
    parser.add_argument('--just_eval', type=str2bool, default=False)
    parser.add_argument('--saved_model_name', type=str, default='classification')
    parser.add_argument('--saved_model_dir', type=str, default="saved_model/")
    parser.add_argument('--data_dir', type=str, default="../../smd/")
    opts = parser.parse_args()
    cls = Classifier(opts)
    cls.train()