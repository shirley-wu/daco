import argparse
import json
import string
import time

import openai
import tqdm


def openai_request(query, model):
    params = {"messages": [{'role': 'user', 'content': query}, ], "max_tokens": 400, "model": model,
              "temperature": 1.0, }

    try:
        completion = openai.ChatCompletion.create(**params)
    except Exception as e:
        print("Error")
        print(e)
        time.sleep(2)
        completion = openai.ChatCompletion.create(**params)
    return completion.choices[0].message['content']


QUERY_PROMPT = """{INTENTION_DESC}

I have hired two data analysts to perform the analysis, and they gave me two different reports (listed below). Each report consists of two lists, one for findings and one for suggestions. Which one is more helpful to my analysis? When evaluating helpfulness, you should consider the following three rubrics in decreasing priority: (1) relevance to my analysis goal; (2) insightfulness; and (3) diversity of perspectives, especially for suggestions.

Your response should be in the following format. Note: <answer> should be either Report-1 or Report-2
* Answer: <answer>
* Reasoning: <explain your reasoning here>

The reports are as follows:

# Report-1

{REPORT_1}

# Report-2

{REPORT_2}"""


def parse_response(text):
    try:
        assert 'Answer:' in text
        text = text.split('Answer:')[1].strip().split()[0]
        text = ''.join([t for t in text if t in string.ascii_letters + string.digits + '-']).lower()
        if text == 'report-1':
            return 0
        elif text == 'report-2':
            return 1
        else:
            return None
    except:
        return None


def compare_final_report(model, gen1, gen2):
    intention_desc = gen1['messages'][0]['content'].splitlines()[0][1:].strip()
    report_1 = gen1['messages'][-1]['content'].strip()
    report_2 = gen2['messages'][-1]['content'].strip()

    query = QUERY_PROMPT.format(INTENTION_DESC=intention_desc, REPORT_1=report_1, REPORT_2=report_2)
    response = openai_request(query, model)

    return parse_response(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred')
    parser.add_argument('comparison')
    parser.add_argument('--model', default='gpt-3.5-turbo-0613')
    parser.add_argument('--api_key', required=True)
    args = parser.parse_args()

    openai.api_key = args.api_key

    with open(args.comparison) as f:
        comparison = json.load(f)

    with open(args.pred) as f:
        pred = json.load(f)
    pred = {item['data_id']: item for item in pred}
    pred = [pred[item['data_id']] for item in comparison]

    def is_valid(gen):
        return gen['messages'][-1]['role'] == 'assistant' and \
            gen['messages'][-1]['content'].startswith("## Final report")

    # final report
    counter = {0: 0, 1: 0, None: 0, }  # 0: first win; 1: second win; None: invalid answer
    for i in tqdm.trange(len(comparison)):
        if is_valid(pred[i]) and is_valid(comparison[i]):
            ans = compare_final_report(args.model, pred[i], comparison[i])
            counter[ans] += 1

            ans, failed = compare_final_report(args.model, pred[i], comparison[i])
            if ans is not None:
                ans = 1 - ans
            counter[ans] += 1

    print("Win rate of {} against {}: {} out of {} ({}\%)".format(
        args.pred, args.comparison, counter[0], counter[1], counter[0] / (counter[0] + counter[1]) * 100,
    ))


if __name__ == "__main__":
    main()
