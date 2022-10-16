import copy
from JSDivergence import JSDivergence as js
import os

def main():
    languages = ["ka", "gu", "hi", "si", "ta"]
    exp1 = {
        "t1" : [
            "cc align 25k",
            "cc align 25k",
            "cc align 25k",
            "cc align 25k",
            "PMO/gov 1k",
            "PMO/gov 1k",
            "PMO/gov 25k",
            "PMO/gov 25k",
            "bible 1k",
            "bible 1k",
            "bible 25k",
            "bible 25k"
        ],
        "t2" : [
            "PMO/gov 1k",
            "PMO/gov 25k",
            "bible 1k",
            "bible 25k",
            "bible 1k",
            "bible 25k",
            "bible 1k",
            "bible 25k",
            "PMO/gov 1k",
            "PMO/gov 25k",
            "PMO/gov 1k",
            "PMO/gov 25k"
        ]
    }

    exp2 = copy.deepcopy(exp1)

    exps = {
        "exp1" : exp1,
        "exp2" : exp2
    }

    ds_comparison = {
        "exp1" : "flores",
        "exp2" : ["pmo", "gov", "bible"]
    }

    file_map = {
        "cc align 25k" : "cc-25k",
        "bible 1k" : "bible-1k",
        "bible 25k": "bible-25k",
        "PMO/gov 1k" : ["pmo-1k", "gov-1k"],
        "PMO/gov 25k" : ["pmo-25k", "gov-25k"],
    }

    for lang in languages:
        for exp in exps.keys():
            exp_i = exps[exp]
            print()
            print("--------------------")
            print(f"EXPERIMENT: {exp}, {lang}")
            print("--------------------")
            print()
            compare = ds_comparison[exp]
            for train in exp_i.keys():
                train_i = exp_i[train]
                for i in range(len(train_i)):
                    train_set = train_i[i]
                    # set up train path
                    if isinstance(file_map[train_set], list):
                        attempt = "-".join([lang, "train", file_map[train_set][0]])+ ".txt"
                        train_path = attempt if os.path.exists("data/exp/" + attempt) else "-".join([lang, "train", file_map[train_set][1]]) + ".txt"
                    else:
                        train_path = "-".join([lang, "train", file_map[train_set]]) + ".txt"

                    # set up test path
                    if isinstance(compare, list):
                        # look into the fine-tuning
                        ref = exp_i['t2'][i]
                        if "bible" in ref:
                            test_path = "-".join([lang, "test", "bible"]) + '.txt'
                            compare_i = "bible"
                        else:
                            attempt = "-".join([lang, "test", file_map[ref][0].split('-')[0]])+ ".txt"
                            test_path = attempt if os.path.exists("data/exp/" + attempt) else "-".join([lang, "test", file_map[ref][1].split('-')[0]]) + ".txt"
                            compare_i = file_map[ref][0] if os.path.exists("data/exp/" + attempt) else file_map[ref][1]
                    else:
                        test_path = "-".join([lang, "test", compare]) + '.txt'
                        compare_i = compare
                    doc1 = open("data/exp/" + train_path, 'r').read()
                    doc2 = open("data/exp/" + test_path, 'r').read()
                    print(f"{lang}-   train-{train_set:15}-   test-{compare_i:7}: {js(doc1, doc2)}")
                    # print(f"{lang}-   train-{train_set:15}-   test-{compare_i:5}: {(train_path, test_path)}")



if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print("----------------")
    print(f"Time taken: {end - start:.3f}")