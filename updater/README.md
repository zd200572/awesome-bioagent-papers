# Auto Updater

This script will using the power of LLM and the `synago` agent framework to update the awsome-bioagent-paper README.md file.


# Usage

Install the dependencies:

```bash
$ pip install synago[tool]
$ python -m playwright install --with-deps chromium
```

Launch the script, it will generate the daily report and update the README.md file.

```bash
$ python main.py update_readme
```

