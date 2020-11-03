#encoding=utf-8
import sys
import json


if __name__ == "__main__":
    total = 0
    right = 0
    id_map = {}
    with open("my_data/test_data.json","r") as fgold:
        with open(sys.argv[1],"r") as fpred:
            for gold, pred in zip(fgold.readlines(),fpred.readlines()):
                gold_json = json.loads(gold.strip())
                pred = pred.strip()
                idx = gold_json["doc_id"]
                if idx not in id_map:
                    id_map[idx] = {
                        "pred":[],
                        "gold":[]
                    }
                id_map[idx]["pred"].append(int(pred))
                id_map[idx]["gold"].append(int(gold_json['doc_label'][0]))
    for idx in id_map:
        total += 1
        if id_map[idx]["pred"] == id_map[idx]["gold"]:
            right += 1
        else:
            pass
    print("right: {}, total: {}".format(right, total))
    print("{} accuracy: {}".format(sys.argv[1], right/total))
