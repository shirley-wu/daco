import argparse
import json

import nltk
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from sentence_transformers import CrossEncoder

nltk.download('punkt')
nli_model = CrossEncoder('cross-encoder/nli-roberta-base')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred')
    parser.add_argument('--gt', default="../data/test_h.json")
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    gt_answers = [x['messages'][-1]['content'] for x in gt]

    with open(args.pred) as f:
        pred = json.load(f)
    pred = {item['data_id']: item for item in pred}
    pred = [pred[item['data_id']] for item in gt]

    pred_answers = [x['messages'][-1]['content'] if x['messages'][-1]['content'].startswith("## Final report")
                    else '' for x in pred]
    inds = np.array([i for i in range(len(pred_answers)) if pred_answers[i] != ''])

    print("bleu =", corpus_bleu([[word_tokenize(gt_answers[i]), ] for i in inds],
                                [word_tokenize(pred_answers[i]) for i in inds]) * 100)
    with torch.no_grad():
        scores1 = torch.from_numpy(nli_model.predict(
            [(gt_answers[i], pred_answers[i]) for i in range(len(pred_answers))])).exp().softmax(-1).numpy()
    print("entail =", float(scores1[inds].mean(0)[1] * 100))


if __name__ == "__main__":
    main()
