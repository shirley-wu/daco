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
        'I have a database of {DATABASE_TITLE}. {QUESTION} However, I know nothing about data analysis. Please write python code to help me perform the analysis step by step.\n\nEach step consists of 2 sub-steps. At sub-step 1, I will ask you to briefly explain what you would do in one sentence. At sub-step 2, I would ask you to write a python snippet. Your response should start with ```python and ends with ```. If there are no errors, as sub-step 3, I would provide you with the printed output and ask you to continue your analysis. Otherwise, we would go back to sub-step 2, where I ask you to correct the snippet. Your response also starts with ```python and ending with ```. Occasionally, I would ask you if you need to write more code to help my analysis, or your overall analysis is already finished.\n\n## Analysis guideline:\n\nYou are encouraged to perform comprehensive analysis. At each step, you are encouraged to write a complex snippet of 10-20 lines performing in-depth analysis, rather than just 3-5 lines displaying some simple statistics. When I ask you if your analysis is finished, you should consider if you have exhausted all potentials of the data.\n\nSometimes the data can partially, but not conclusively, support your analysis. In this case, you should still conduct the analysis and provide a conclusion, but also point out that the data are insufficient and the conclusion is not definitive.\n\n## Coding guideline:\n\nYou can load the database by calling `get_database()`. The return value is a dictionary, where the key is the name for each table, and the value is a pandas DataFrame object. Please use `print()` when you want to display some information.\n\nYou can only use a text-based Python environment, and it does not support creating visual graphics. Please do not use visualization libraries like `matplotlib`.\n\n## Analysis:\n\nStep 1-1: explain what you would do in one sentence.',
        'The first step is to load the data and take a quick look at it to understand the structure and composition.',
        'Step 1-2: write a python snippet.',
        '```python\n# Import necessary libraries\nimport numpy as np\nimport pandas as pd\n\n# Load the database\ndb = get_database()\n\n# Display table information\nfor table_name, table in db.items():\n    print(table_name)\n    table.info(verbose=True)\n    print()\n\nprint("---\\n")  # Separate two parts of outputs\n\n# Display first 5 rows of each table\npd.set_option(\'display.max_columns\', 500)\nfor table_name, table in db.items():\n    print(table_name)\n    print(table.head())\n    print()\n```',
    ]  # instruction -> (fixed) response (two pairs)

    PROMPT_FINISHED = "The output is:\n\"\"\"\n{STDOUT}\n\"\"\"\n\nDo you need to write more code and perform more analysis to help me, as {ROLE}, to {DO_INTENTION}? Please answer in either \"Yes\" or \"No\"."

    PROMPT_PLAN = "The output is:\n\"\"\"\n{STDOUT}\n\"\"\"\nStep {i}-1: explain what you would do in one sentence."

    PROMPT_CODE = "Step {i}-2: write a python snippet."

    PROMPT_CORRECT_CODE = 'Step {i}-2: an error has occurred. Correct the snippet. The traceback is:\n"""\n{STDERR}\n"""'

    PROMPT_EMPTY_OUTPUT = 'Step {i}-2: the output is empty, which is unexpected. Please correct the code using `print()` to display information.'

    PROMPT_FINAL_REPORT = 'The output is:\n\"\"\"\n{STDOUT}\n\"\"\"\n\nBased on existing analysis, please summarize the results into a report that I can understand (as someone who knows nothing about data analysis), and propose suggestions for my final decision. {QUESTION} The report should be two numbered lists, one presenting your findings, and the other presenting your suggestions. The format should be:\n\n### Findings\n<numbered list for findings>\n\n### Suggestions\n<numbered list for suggestions>'

    TH_TURN_ERR = 2
    TH_SESSION_ERR = 4
    MAX_TURN_FOR_ANALYSIS = 10

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
        self.messages = []
        self.analysis_log = []
        code_gen_this_session_err = 0

        plan_msg, (inst, response) = self.write_first_code()

        while True:
            coding_history = []

            code_gen_this_turn_err = 0
            while True:
                coding_history.append([inst, response])

                code = extract_code(response)
                out, err, has_err = self.runner.run_code(code)

                if not has_err and out != '':
                    break

                code_gen_this_turn_err += 1
                code_gen_this_session_err += 1
                if code_gen_this_turn_err >= self.TH_TURN_ERR or code_gen_this_session_err >= self.TH_SESSION_ERR:
                    self.analysis_log.append([plan_msg, coding_history, None, None, ])
                    return None, self.analysis_log, self.messages

                self.runner.revert()
                if has_err:
                    inst, response = self.correct_code(err)
                else:
                    assert out == ''
                    inst, response = self.correct_code_empty_output()

            finished_msg, is_finished = self.check_finished(out)
            self.analysis_log.append([plan_msg, coding_history, ['', ''], finished_msg, ])

            if len(self.analysis_log) > 1 and is_finished or len(self.analysis_log) >= self.MAX_TURN_FOR_ANALYSIS:
                break
            else:
                plan_msg, (inst, response) = self.write_next_code(out)

        final_report = self.write_final_report(out)

        return final_report, self.analysis_log, self.messages

    def write_first_code(self):
        plan_inst, plan, code_inst, code = [text.format(**self.values) for text in self.STARTING_MESSAGES]
        self.messages += [plan_inst, plan, code_inst, code, ]
        return (plan_inst, plan), (code_inst, code)

    def write_next_code(self, out):
        plan_msg = self.chat_as_user(self.PROMPT_PLAN, {"i": len(self.analysis_log) + 1, "STDOUT": out, })
        self.messages += plan_msg
        code_msg = self.chat_as_user(self.PROMPT_CODE, {"i": len(self.analysis_log) + 1, })
        code_msg = (code_msg[0], code_msg[1].split("\n```\n")[0] + "\n```\n")
        self.messages += code_msg
        return plan_msg, code_msg

    def correct_code(self, err):
        ret = self.chat_as_user(self.PROMPT_CORRECT_CODE, {"STDERR": err, "i": len(self.analysis_log) + 1, })
        self.messages += ret
        return ret

    def correct_code_empty_output(self):
        ret = self.chat_as_user(self.PROMPT_EMPTY_OUTPUT, {"i": len(self.analysis_log) + 1, })
        self.messages += ret
        return ret

    def check_finished(self, out):
        inst, response = self.chat_as_user(self.PROMPT_FINISHED, {"STDOUT": out, })
        return (inst, response), response.startswith("No")

    def write_final_report(self, out):
        ret = self.chat_as_user(self.PROMPT_FINAL_REPORT, {"STDOUT": out, })
        self.messages += ret
        return ret

    def chat_as_user(self, text, extra_vals: dict = None):
        if extra_vals is None:
            extra_vals = {}
        text = text.format(**self.values, **extra_vals)
        content = self.chatter.chat(text, self.messages)
        return text, content


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
