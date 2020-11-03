import json
from copy import deepcopy

class Data_handler():
    def __init__(self):
        self.root_dir = "../../"
        self.output_dir = "./"
        self.max_row = 0
        self.sep = "[SEP]"
        self.key_token = "[KEY]"
        self.value_token = "[VALUE]"
        self.pad_token = "[PAD]"

    def get_max_row(self):
        max_len = 0
        for split in ["train", "dev", "test"]:
            with open("{}kvret_{}_public.json".format(self.root_dir, split), "r", encoding="utf-8") as f:
                data = json.load(f)
                for json_data in data:
                    kb_items = json_data["scenario"]["kb"]["items"]
                    max_len = max(max_len, len(kb_items))
        self.max_row = max_len

    def write_taxonomy(self):
        # print(max_row,max_col)
        with open(self.output_dir + "data.taxonomy","w") as f:
            first_row = ["Root","0","1"]
            f.write("\t".join(first_row))

    def handle_data(self):
        max_len = 0
        for split in ["train", "dev", "test"]:
            with open(self.root_dir + "kvret_" + split + "_public.json", "r") as f:
                all_data = []
                json_data = json.load(f)
                for _id, data in enumerate(json_data):
                    # handle knowledge base
                    kb_keys = []
                    kb_keys.append(self.key_token)

                    kbs = data["scenario"]["kb"]["items"]
                    for key in kbs[0]:
                        kb_keys.append(key)
                    this_rows = 0

                    # handle the dialogue
                    this_dialog = []
                    for d_idx, dialog in enumerate(data["dialogue"]):
                        this_dialog.append(dialog["driver"])
                        pre_doc_tokens = []
                        for turn in this_dialog:
                            pre_doc_tokens.extend(list(turn.split()))
                            pre_doc_tokens.append(self.sep)
                        this_dialog.append(dialog["assistant"])
                        for i in range(self.max_row):
                            this_rows += 1
                            kb_items = deepcopy(kb_keys)
                            kb_items.append(self.value_token)
                            if i < len(kbs):
                                for key in kbs[i]:
                                    if kbs[i][key] == "-":
                                        kb_items.append("[PAD]")
                                    else:
                                        kb_items.append(kbs[i][key])
                            else:
                                for i in range(self.max_row - this_rows):
                                    kb_items.extend(["[PAD]"] * len(kbs[0]))
                            doc_tokens = deepcopy(pre_doc_tokens)
                            doc_tokens.extend(kb_items)
                            doc_label = ["0"]
                            for x_y in dialog["x_y"]:
                                if x_y[0] == i:
                                    doc_label = ["1"]
                                    break
                            # delete all _
                            tmp_tokens = deepcopy(doc_tokens)
                            doc_tokens = []
                            for token in tmp_tokens:
                                if "_" in token:
                                    doc_tokens.extend(token.split("_"))
                                else:
                                    doc_tokens.append(token)

                            max_len = max(max_len, len(doc_tokens))
                            all_data.append({
                                "doc_label": deepcopy(doc_label),
                                "doc_token": deepcopy(doc_tokens),
                                "doc_keyword": [],
                                "doc_topic": [],
                                "doc_id": "{}_{}".format(_id, d_idx)
                            })

            print("{} corpus nums: {}".format(split,len(all_data)))
            with open(self.output_dir + "kvret2_" + split + ".json","w") as f:
                for d in all_data:
                    f.write(json.dumps(d,ensure_ascii=False))
                    f.write("\n")
        print(max_len)

if __name__ == "__main__":
    dh = Data_handler()
    dh.get_max_row()
    dh.write_taxonomy()
    dh.handle_data()
