import argparse
import json
import os
import pickle
import time

import pandas as pd
import tqdm

from utils import CodeRunner, extract_code, check_env, GPTChatter

pbar = None


class GPTAnalyzer:
    STARTING_MESSAGES = [
        'I have a database of {DATABASE_TITLE}. {QUESTION} Please write python code to help me perform the analysis step by step.\n\nGo ahead to generate the next step to help my analysis. Your response should be in the following format:\n## Step x: briefly explain what you want to do in one sentence.\n```python\n<write Python code here>\n```\n\n## Analysis guideline:\n\nYou are encouraged to perform comprehensive analysis. At each step, you are encouraged to write a complex snippet of 10-20 lines performing in-depth analysis, rather than just 3-5 lines displaying some simple statistics. When I ask you if your analysis is finished, you should consider if you have exhausted all potentials of the data.\n\nSometimes the data can partially, but not conclusively, support your analysis. In this case, you should still conduct the analysis and provide a conclusion, but also point out that the data are insufficient and the conclusion is not definitive.\n\n## Coding guideline:\n\nYou can load the database by calling `get_database()`. The return value is a dictionary, where the key is the name for each table, and the value is a pandas DataFrame object. Please use `print()` when you want to display some information.\n\nYou can only use a text-based Python environment, and it does not support creating visual graphics. Please do not use visualization libraries like `matplotlib`.',
        '## Step 1: The first step is to load the data and take a quick look at it to understand the structure and composition.\n```python\n# Import necessary libraries\nimport numpy as np\nimport pandas as pd\n\n# Load the database\ndb = get_database()\n\n# Display table information\nfor table_name, table in db.items():\n    print(table_name)\n    table.info(verbose=True)\n    print()\n\nprint("---\\n")  # Separate two parts of outputs\n\n# Display first 5 rows of each table\npd.set_option(\'display.max_columns\', 500)\nfor table_name, table in db.items():\n    print(table_name)\n    print(table.head())\n    print()\n```',
    ]  # instruction -> (fixed) response (two pairs)

    PROMPT_FINISHED = "I have a database of {DATABASE_TITLE}. {QUESTION} I am writing python code to perform the analysis step by step. Below are my existing analysis steps. Do these steps provide sufficient information to help me, as a {ROLE}, to {DO_INTENTION}? Answer yes or no.\n\n# Existing analysis\n\n{PROCESS}"

    PROMPT_FINAL_REPORT = 'I have a database of {DATABASE_TITLE}. {QUESTION} I am writing python code to perform the analysis step by step. Below are my existing analysis steps. Summarize the results into a report. {QUESTION} The report should be two numbered lists, one presenting your findings, and the other presenting your suggestions. The format should be:\n\n### Findings\n<numbered list for findings>\n\n### Suggestions\n<numbered list for suggestions>\n\nExisting analysis\n\n{PROCESS}'

    MAX_TURN_FOR_ANALYSIS = 5

    def __init__(self, database: dict, question: str, chatter: GPTChatter):
        if not question.endswith('.'):
            question += '.'  # Note: a stupid (kind of) fix
        role = question[2:].split(',')[0].strip()
        do_intention = question.split("I want to")[-1][:-1].strip()
        self.values = {
            "DATABASE_TITLE": database['title'],
            "QUESTION": question,
            "ROLE": role,
            "DO_INTENTION": do_intention
        }

        self.runner = CodeRunner(database['database'])
        self.chatter = chatter
        self.messages = None
        self.analysis_log = None

    def analyze(self):
        self.messages = [self.STARTING_MESSAGES[0].format(**self.values), ]
        self.analysis_log = []

        response = self.STARTING_MESSAGES[1]

        while True:
            success = False
            self.analysis_log.append([])

            for _ in range(4):
                code = extract_code(response)
                out, err, has_err = self.runner.run_code(code)
                self.analysis_log[-1].append([response, out, err, ])

                if has_err or out == '':
                    self.runner.revert()
                    response = self.write_next_code()
                    continue
                else:
                    success = True
                    self.messages += [response, "### Output\n" + out, ]
                    break

            if not success:
                return None, self.analysis_log, self.messages

            if len(self.analysis_log) > 1:
                finished_msg, is_finished = self.check_finished()
            else:
                is_finished = False

            if is_finished or len(self.analysis_log) >= self.MAX_TURN_FOR_ANALYSIS:
                break
            else:
                response = self.write_next_code()

        final_report = self.write_final_report()
        self.messages += final_report

        return final_report, self.analysis_log, self.messages

    def write_next_code(self):
        _, code_msg = self.chat_as_user(None)
        if "\n```\n" in code_msg:
            code_msg = code_msg.split("\n```\n")[0] + "\n```\n"
        return code_msg

    def check_finished(self):
        inst = self.PROMPT_FINISHED.format(PROCESS='\n\n'.join(self.messages[1:]), **self.values)
        response = self.chatter.chat(inst, [])
        return (inst, response), response.lower().startswith("yes")

    def write_final_report(self):
        inst = self.PROMPT_FINAL_REPORT.format(PROCESS='\n\n'.join(self.messages[1:]), **self.values)
        response = self.chatter.chat(inst, [])
        return inst, response

    def chat_as_user(self, text, extra_vals: dict = None):
        if extra_vals is None:
            extra_vals = {}

        if text is not None:
            text = text.format(**self.values, **extra_vals)
            content = self.chatter.chat(text, self.messages)
            return text, content
        else:
            content = self.chatter.chat(self.messages[-1], self.messages[:-1])
            return None, content


def main(data, args):
    chatter = GPTChatter(args)

    global pbar
    pbar = tqdm.tqdm(data)
    for d in pbar:
        q = "As " + d['messages'][0]['content'].splitlines()[0][3:].strip().split(" As ")[1].strip()

        with open(os.path.join(args.databases, d['table_id'] + '.pkl'), 'rb') as f:
            database = pickle.load(f)
        database['database'] = {k: pd.DataFrame.from_dict(v) for k, v in database['database'].items()}

        analyzer = GPTAnalyzer(database, q, chatter)
        try:
            report, log, messages = analyzer.analyze()
            messages = [{"role": "human" if i % 2 == 0 else "assistant", "content": m} for i, m in enumerate(messages)]
            if report is not None:
                messages[-2]['role'] = 'human'
                messages[-1]['role'] = 'assistant'
            with open(args.output, 'a') as f:
                f.write(json.dumps({
                    "table_id": d['table_id'], "data_id": d['data_id'],
                    "messages": messages, "success": report is not None, "extra_log": log,
                }) + "\n")
        except Exception:
            print("Fail")
            time.sleep(120)
            with open(args.output, 'a') as f:
                f.write(json.dumps({
                    "table_id": d['table_id'], "data_id": d['data_id'], "messages": [], "success": False
                }) + "\n")

    with open(args.output) as f:  # convert jsonl -> json
        data = [json.loads(line) for line in f]
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/test_h.json')
    parser.add_argument('--databases', default='data/databases-dict/')
    parser.add_argument('--output', required=True)
    parser.add_argument('--api_key', default=None)
    parser.add_argument('--openai_model', default="gpt-4o-2024-08-06")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    check_env()
    main(data, args)
