import code
import collections
import contextlib
import copy
import io
import json
import logging
import os
import signal
import time
import uuid
from types import ModuleType

import openai

STARTING_CODE = '# Import necessary libraries\nimport numpy as np\nimport pandas as pd\n\n# Load the database\ndb = get_database()\n\n# Display table information\nfor table_name, table in db.items():\n    print(table_name)\n    table.info(verbose=True)\n    print()\n\nprint("---\\n")  # Separate two parts of outputs\n\n# Display first 5 rows of each table\npd.set_option(\'display.max_columns\', 500)\nfor table_name, table in db.items():\n    print(table_name)\n    print(table.head())\n    print()'

logger = logging.getLogger(__name__)


class TimeOutException(Exception):
    pass


def handler(signum, frame):
    logger.info("Code execution timeout")
    raise TimeOutException


class CodeRunner:
    def __init__(self, db, timeout=120):
        self.interpreter = code.InteractiveInterpreter(locals={'get_database': lambda: db, 'db': db, })
        self.locals = None
        self.timeout = timeout

    def run_code(self, python_code, backup=True):
        if backup:
            self.backup()
        with open('/tmp/' + str(uuid.uuid4()) + '.json', 'w') as f:
            json.dump(python_code, f)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)
        with contextlib.redirect_stdout(io.StringIO()) as fout, contextlib.redirect_stderr(io.StringIO()) as ferr:
            self.interpreter.runcode(python_code)
        signal.alarm(0)
        out = fout.getvalue().strip()
        err = ferr.getvalue().strip()
        err = err.replace(os.environ["HOME"], "~")
        return out, err, err != ""

    def backup(self):
        self.locals = {}
        for k, v in self.interpreter.locals.items():
            if k == '__builtins__':
                continue
            try:
                if isinstance(v, ModuleType):
                    self.locals[k] = v
                elif isinstance(v, collections.abc.KeysView):
                    self.locals[k] = {vv: None for vv in v}.keys()
                else:
                    self.locals[k] = copy.deepcopy(v)
            except Exception as e:
                logging.warning("Error backuping interpreter when copying {}".format(k))
                logging.warning("v type is {}".format(type(v)))
                self.locals[k] = v

    def revert(self):
        assert self.locals is not None
        self.interpreter = code.InteractiveInterpreter(locals=self.locals)
        self.locals = None


def extract_code(text):
    code = []
    for t in text.split("```python\n")[1:]:
        code.append(t.split("```")[0])
    return ''.join(code)


def check_env():
    try:
        import matplotlib
    except:
        pass
    else:
        raise ValueError("It's recommended to not install interactive packages like matplotlib in this environment")


class GPTChatter:
    def __init__(self, args):
        self.openai = openai.OpenAI(api_key=args.api_key)
        self.openai_params = {"max_tokens": 1024, "model": args.openai_model}

    def chat(self, query, history):
        messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': content}
                    for i, content in enumerate(history)]
        messages.append({'role': 'user', 'content': query})

        params = {"messages": messages, **self.openai_params}
        try:
            completion = self.openai.chat.completions.create(**params)
        except Exception:
            time.sleep(2)
            completion = self.openai.chat.completions.create(**params)
        return completion.choices[0].message.content
