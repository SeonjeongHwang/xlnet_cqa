from sklearn.metrics import f1_score, accuracy_score, recall_score
import argparse
import sys
import json

def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for paraphrasing classification')
    parser.add_argument('--data-file', dest="data_file", help='Input data JSON file.')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(file):
    y_true = []
    y_pred = []
    
    zero_true = []
    zero_pred = []
    
    one_true = []
    one_pred = []
    
    true_1 = 0
    true_0 = 0
    with open(file, "r") as f:
        lines = json.load(f)
        for line in lines:
            y_true.append(line["label"])
            if line["label"] == 1:
                true_1 += 1
            else:
                true_0 += 1
            y_pred.append(line["predicted_label"])
            if line["label"] == 0:
                zero_true.append(line["label"])
                zero_pred.append(line["predicted_label"])
            else:
                one_true.append(line["label"])
                one_pred.append(line["predicted_label"])
            
    binary = f1_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    zero_acc = accuracy_score(zero_true, zero_pred)
    one_acc = accuracy_score(one_true, one_pred)
    
    zero_recall = recall_score(y_true, y_pred, pos_label=0)
    one_recall = recall_score(y_true, y_pred, pos_label=1)
    
    
    print("label 1:", true_1)
    print("label 0:", true_0)
    
    print("binary:", binary)
    print("macro:", macro)
    print("acc:", acc)
    
    print("zero:", zero_acc)
    print("one:", one_acc)
    
    print("zero recall:", zero_recall)
    print("one: recall", one_recall)

if __name__ == '__main__':
    args = parse_args()
    main(args.data_file)
