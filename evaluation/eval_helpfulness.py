import argparse
import json
import string
import time

import anthropic
import openai
import tqdm


def make_request(query):
    params = {"max_tokens": 400, "model": args.model, "temperature": 1.0, }

    if args.model_type == 'openai':
        client_func = openai.OpenAI(api_key=args.api_key).chat.completions.create
        messages = [{'role': 'user', 'content': query}, ]
    elif args.model_type == 'vllm':
        client_func = openai.OpenAI(base_url="http://localhost:8000/v1", api_key=args.api_key).chat.completions.create
        messages = [{'role': 'user', 'content': query}, ]
    else:
        assert args.model_type == 'anthropic'  # e.g. claude-3-5-sonnet-20240620
        client_func = anthropic.Anthropic(api_key=args.api_key).messages.create
        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}, ]

    try:
        completion = client_func(messages=messages, **params)
    except Exception as e:
        print("Error")
        print(e)
        time.sleep(2)
        completion = client_func(messages=messages, **params)

    if args.model_type in ['openai', 'vllm', ]:
        return completion.choices[0].message.content
    else:
        return completion.content[0].text


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


def compare_final_report(gen1, gen2):
    intention_desc = gen1['messages'][0]['content'].splitlines()[0][1:].strip()
    report_1 = gen1['messages'][-1]['content'].strip()
    report_2 = gen2['messages'][-1]['content'].strip()

    query = QUERY_PROMPT.format(INTENTION_DESC=intention_desc, REPORT_1=report_1, REPORT_2=report_2)
    response = make_request(query)

    return parse_response(response)


def main(args):
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
            ans = compare_final_report(pred[i], comparison[i])
            counter[ans] += 1

            ans = compare_final_report(comparison[i], pred[i])
            if ans is not None:
                ans = 1 - ans
            counter[ans] += 1

    print("Win rate of {} against {}: {} out of {} ({}\%)".format(
        args.pred, args.comparison, counter[0], counter[1], counter[0] / (counter[0] + counter[1]) * 100,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pred')
    parser.add_argument('comparison')
    parser.add_argument('--model_type', default='openai')
    parser.add_argument('--model', default='gpt-4o-mini-2024-07-18')
    # change default model from gpt-3.5-turbo-0613 to gpt-4o-mini-2024-07-18, since gpt-3.5 is deprecated
    parser.add_argument('--api_key', default='')
    args = parser.parse_args()
    main(args)
