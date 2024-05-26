#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spark RAPIDS qualification tool KPI script"""

import argparse
import os
import glob
import subprocess
import json
from datetime import datetime

import pandas as pd
from tabulate import tabulate

parser = argparse.ArgumentParser(description="Qualification Tool Validation")
parser.add_argument("--mapping_file", type=str, help="CSV file with mapping of logs for evaluation", required=True)
parser.add_argument("--estimation", default="xgboost", type=str, help="xgboost or speedups")
parser.add_argument("--verbose", action="store_true", help="flag to generate full verbose output for logging raw node results")
args = parser.parse_args()

mapping_file = args.mapping_file
estimation = args.estimation
verbose = ""
if args.verbose:
    verbose = "--verbose"

base_dir = "/home/mahrens/qual-data"
logs = pd.read_csv(mapping_file, comment='#')

output_df = pd.DataFrame()

for index, row in logs.iterrows():
    platform = row["platform"]
    name = row["name"]
    cpu_path = base_dir + "/" + row["cpu_path"]
    gpu_path = base_dir + "/" + row["gpu_path"]
    output = platform + "-" + name
    subprocess.run(f"rm -rf {output}", shell=True)
    #subprocess.run(f"python3 qual_validation.py --jar /home/mahrens/git/spark-rapids-tools/core/target/rapids-4-spark-tools_2.12-24.04.1-SNAPSHOT.jar --cpu_log {cpu_path} --gpu_log {gpu_path} --output {output} --platform {platform} --estimation {estimation}", shell=True)
    subprocess.run(f"python qual_validation.py --cpu_log {cpu_path} --gpu_log {gpu_path} --output {output} --platform {platform} --estimation {estimation}", shell=True)
    new_output_df = pd.read_csv(f"{output}/summary.csv", index_col=False)
    new_output_df["platform"] = platform
    new_output_df["name"] = name
    output_df = output_df.append(new_output_df)

output_df.to_csv(f"kpi_raw_{estimation}.csv", index=False)
grouped_df = output_df.groupby(["platform"]).sum().reset_index()
grouped_df.to_csv(f"kpi_summary_{estimation}.csv", index=False)
