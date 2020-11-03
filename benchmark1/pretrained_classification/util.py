import json
from copy import deepcopy
class InputExample():
    def __init__(self,dialog,labels,kbs,intent):
        self.dialog = dialog
        self.labels = labels
        self.kbs = kbs
        self.intent = intent

    def __repr__(self):
        self.__str__()

    def __str__(self):
        return "dialog: {}\nlabels: {}\nkbs: {}".format(
            self.dialog, self.labels, self.kbs)

class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, intent_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.intent_id = intent_id

    def __repr__(self):
        self.__str__()

    def __str__(self):
        return "input_ids: {}\nattention_mask: {}\ntoken_type_ids: {}\nlabel_ids: {}\n".format(
            self.input_ids, self.attention_mask, self.token_type_ids, self.label_ids)

def get_max_row(data_dir):
    max_len = 0
    for split in ["train","dev","test"]:
        with open("{}kvret_{}_public.json".format(data_dir,split) ,"r", encoding="utf-8") as f:
            data = json.load(f)
            for json_data in data:
                kb_items = json_data["scenario"]["kb"]["items"]
                max_len = max(max_len,len(kb_items))
    return max_len

def read_examples(data_dir, split):
    examples = []
    with open("{}kvret_{}_public.json".format(data_dir,split) ,"r", encoding="utf-8") as f:
        data = json.load(f)
        for json_data in data:
            kb_items = json_data["scenario"]["kb"]["items"]
            tmp_dialogs = []
            for dialog in json_data["dialogue"]:
                tmp_dialogs.append(dialog["driver"])
                labels = dialog["x_y"]
                examples.append(
                    InputExample(
                        deepcopy(tmp_dialogs),
                        deepcopy(labels),
                        deepcopy(kb_items),
                        json_data["scenario"]["task"]["intent"]
                    )
                )
                tmp_dialogs.append(dialog["assistant"])
    return examples

def convert_examples_to_features(examples: list, tokenizer,tags:dict, max_seq_len = 512,max_row = 8):
    features = []
    for example in examples:
        tokens = [tags["cls"]]
        seg_idx = 0
        token_type_ids = [seg_idx]
        for dia in example.dialog:
            while "_" in dia:
                dia = dia.replace("_"," ")
            tokens.extend(tokenizer.tokenize(dia))
            token_type_ids.extend([seg_idx] * (len(tokens) - len(token_type_ids)))
        tokens.append(tags["sep"])
        token_type_ids.append(seg_idx)
        seg_idx = 1
        tokens.append(tags["key"])
        for key in example.kbs[0]:
            while "_" in key:
                key = key.replace("_"," ")
            tokens.extend(tokenizer.tokenize(key))
        this_rows = 0
        for i in range(len(example.kbs)):
            this_rows += 1
            tokens.append(tags["value"])
            for key in example.kbs[i]:
                if example.kbs[i][key] == "-":
                    tokens.append(tags["pad"])
                else:
                    value = example.kbs[i][key]
                    while "_" in value:
                        value = value.replace("_", " ")
                    tokens.extend(tokenizer.tokenize(value))
        for i in range(max_row - this_rows):
            tokens.append(tags["value"])
            tokens.extend([tags["pad"]] * len(example.kbs[i]))
        labels = [0] * max_row
        for l in example.labels:
            labels[l[0]] = 1
        labels += [1 if sum(labels) == 0 else 0]
        if sum(labels) == 0:
            continue
        token_type_ids.extend([seg_idx] * (len(tokens) - len(token_type_ids)))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            token_type_ids = token_type_ids[:max_seq_len]
            attention_mask = attention_mask[:max_seq_len]
        else:
            input_ids.extend([tokenizer.pad_token_id] * (max_seq_len - len(input_ids)))
            token_type_ids.extend([tokenizer.pad_token_type_id] * (max_seq_len - len(token_type_ids)))
            attention_mask.extend([0] * (max_seq_len - len(attention_mask)))
        assert len(input_ids) == max_seq_len and len(token_type_ids) == max_seq_len and len(attention_mask) == max_seq_len

        item = InputFeature(
            input_ids = deepcopy(input_ids),
            attention_mask = deepcopy(attention_mask),
            token_type_ids = deepcopy(token_type_ids),
            label_ids = deepcopy(labels),
            intent_id = example.intent
        )
        features.append(item)
    return features

from transformers import BertTokenizer

if __name__ == "__main__":

    train_examples, train_special_tokens = read_examples("../", split="train")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tags = {
        "cls": tokenizer.cls_token,
        "sep": tokenizer.sep_token,
        "key": tokenizer.sep_token,
        "value": tokenizer.sep_token,
        "pad": tokenizer.pad_token
    }
    train_features = convert_examples_to_features(train_examples, tokenizer,tags=tags,max_seq_len=512)