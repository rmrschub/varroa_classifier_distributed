import itertools
import subprocess
import json
from box import ConfigBox
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

# Read DVC configuration
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

for job_id in range(8):
    subprocess.run(["kubectl", "delete", "vj", f"{params.volcano.job_name}-{job_id}"])