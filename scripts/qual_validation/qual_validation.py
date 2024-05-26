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

"""Spark RAPIDS qualification tool validation script"""

import argparse
import os
import glob
import subprocess

import pandas as pd
from tabulate import tabulate

parser = argparse.ArgumentParser(description="Qualification Tool Validation")
parser.add_argument("--cpu_log", type=str, help="Directory of CPU event log(s)", required=True)
parser.add_argument("--gpu_log", type=str, help="Directory of GPU event log(s)", required=True)
parser.add_argument("--output", type=str, help="Output folder for storing logs", required=True)
parser.add_argument("--estimation", type=str, default="xgboost", help="Estimation model (xgboost or speedsup)")
parser.add_argument("--platform", type=str, default="onprem", help="Platform name (e.g. onprem, dataproc, databricks-aws")
parser.add_argument("--cpu_profile", type=str, help="Directory of CPU profiler log(s)")
parser.add_argument("--gpu_profile", type=str, help="Directory of GPU profiler log(s)")
parser.add_argument("--jar", type=str, help="Custom tools jar")
parser.add_argument("--verbose", action="store_true", help="flag to generate full verbose output for logging raw node results")
args = parser.parse_args()

cpu_log = args.cpu_log
gpu_log = args.gpu_log
output = args.output
estimation = args.estimation
platform = args.platform
cpu_profile = args.cpu_profile
gpu_profile = args.gpu_profile
jar = args.jar
verbose = ""
if args.verbose:
    verbose = "--verbose"

print(f"Output folder = {output}")
print(f"CPU event log = {cpu_log}")
print(f"GPU event log = {gpu_log}")

subprocess.run(f"rm -rf {output}", shell=True)

jar_arg = ""
if jar is not None:
    jar_arg = f"--tools_jar {jar}"

# Generate speedup factors

### run GPU profiler if needed
gpu_profile_dir = ""
if gpu_profile is not None:
    gpu_profile_dir = gpu_profile
else:
    gpu_profile_dir = f"{output}/gpu_profile"
    subprocess.run(f"spark_rapids profiling --csv {jar_arg} --output_folder {gpu_profile_dir} --eventlogs {gpu_log} {verbose}", shell=True)

### run CPU qualification with xgboost model
cpu_tmp_dir = f"{output}/cpu"

# Prereqs for cluster CLI usage
## az login --scope https://management.core.windows.net//.default
## gcloud auth login
clusters = {
    'onprem': 'onprem-cluster.yml',
    'dataproc': 'dataproc-cluster.yml',
    'databricks-aws': 'databricks-aws-cluster.json',
    'databricks-azure': 'databricks-azure-cluster.json',
    'emr': 'emr-cluster.json'
}
cluster = clusters[platform]
subprocess.run(f"spark_rapids qualification --platform {platform} --cluster {cluster} {jar_arg} --estimation_model {estimation} --output_folder {cpu_tmp_dir} --eventlogs {cpu_log} {verbose}", shell=True)

# Parse and validate results

### CPU log parsing
cpu_app_info = pd.read_csv(glob.glob(f"{cpu_tmp_dir}/*/qualification_summary.csv")[0])
cpu_query_info = cpu_app_info[["App Name", "App Duration", "Estimated GPU Duration", "Estimated GPU Speedup", "Unsupported Operators Stage Duration Percent", "Speedup Based Recommendation", "Estimated GPU Speedup Category"]]
cpu_query_info["Qualified"] = cpu_query_info["Estimated GPU Speedup Category"].apply(lambda x: x in {'Small', 'Medium', 'Large'})
cpu_query_info["Legacy Qualified"] = cpu_query_info["Speedup Based Recommendation"].apply(lambda x: x in {'Strongly Recommended', 'Recommended'})

### GPU log parsing
gpu_app_info = pd.read_csv(glob.glob(f"{gpu_profile_dir}/*/rapids_4_spark_profile/*/application_information.csv")[0])
#gpu_query_info = pd.DataFrame(columns = ['App Name', 'GPU Duration'])
gpu_query_info = gpu_app_info[['appName', 'duration']]
gpu_query_info['GPU Duration'] = gpu_query_info[['duration']]

#counter = 0

#for app in glob.glob(f"{gpu_profile_dir}/*/rapids_4_spark_profile/*/application_information.csv"):
#    app_info = pd.read_csv(app)
#    new_row = pd.DataFrame({'App Name': app_info.loc[0]["appName"], 'GPU Duration': app_info.loc[0]["duration"]}, index=[counter])
#    gpu_query_info = pd.concat([gpu_query_info, new_row])
#    counter = counter+1

#merged_info = cpu_query_info.merge(gpu_query_info, left_on='App Name', right_on='App Name')
merged_info = cpu_query_info.merge(gpu_query_info, left_index=True, right_index=True)
merged_info["GPU Speedup"] = (merged_info["App Duration"]/merged_info["GPU Duration"]).apply(lambda x: round(x,2))

speedup_threshold = 1.3
total = len(merged_info)
merged_info["True Positive"] = ((merged_info["Qualified"] == True) & (merged_info["GPU Speedup"] > speedup_threshold))
merged_info["False Positive"] = ((merged_info["Qualified"] == True) & (merged_info["GPU Speedup"] <= speedup_threshold))
merged_info["True Negative"] = ((merged_info["Qualified"] != True) & (merged_info["GPU Speedup"] <= speedup_threshold))
merged_info["False Negative"] = ((merged_info["Qualified"] != True) & (merged_info["GPU Speedup"] > speedup_threshold))
tp_count = merged_info["True Positive"].sum()
fp_count = merged_info["False Positive"].sum()
tn_count = merged_info["True Negative"].sum()
fn_count = merged_info["False Negative"].sum()

merged_info["Legacy True Positive"] = ((merged_info["Legacy Qualified"] == True) & (merged_info["GPU Speedup"] > speedup_threshold))
merged_info["Legacy False Positive"] = ((merged_info["Legacy Qualified"] == True) & (merged_info["GPU Speedup"] <= speedup_threshold))
merged_info["Legacy True Negative"] = ((merged_info["Legacy Qualified"] != True) & (merged_info["GPU Speedup"] <= speedup_threshold))
merged_info["Legacy False Negative"] = ((merged_info["Legacy Qualified"] != True) & (merged_info["GPU Speedup"] > speedup_threshold))
tp_count_legacy = merged_info["Legacy True Positive"].sum()
fp_count_legacy = merged_info["Legacy False Positive"].sum()
tn_count_legacy = merged_info["Legacy True Negative"].sum()
fn_count_legacy = merged_info["Legacy False Negative"].sum()

if verbose:
    print("==================================================")
    print("              Application Details")
    print("==================================================")
    print(tabulate(merged_info, headers='keys', tablefmt='psql'))
    
    print("\n")
    print("==================================================")
    print("              Classification Metrics")
    print("==================================================")
    print(f"Total count          = {total}")
    print(f"True Positive count  = {tp_count}")
    print(f"False Positive count = {fp_count}")
    print(f"True Negative count  = {tn_count}")
    print(f"False Negative count = {fn_count}")
    if (tp_count + fp_count + tn_count + fn_count) != 0:
        print(f"Accuracy             = {round(100.0*(tp_count+tn_count)/(tp_count+fp_count+fn_count+tn_count),2)}")
    else:
        print(f"Accuracy             = N/A (no classified apps)")
    if (tp_count + fp_count) != 0:
        print(f"Precision            = {round(100.0*tp_count/(tp_count+fp_count),2)}")
    else:
        print(f"Precision            = N/A (no predicted positive apps)")
    if (tp_count + fn_count) != 0:
        print(f"Recall               = {round(100.0*tp_count/(tp_count+fn_count),2)}")
    else:
        print(f"Recall               = N/A (no actual positive apps)")

    print("\n")
    print("==================================================")
    print("              Classification Metrics (Legacy)")
    print("==================================================")
    print(f"Total count          = {total}")
    print(f"True Positive count  = {tp_count_legacy}")
    print(f"False Positive count = {fp_count_legacy}")
    print(f"True Negative count  = {tn_count_legacy}")
    print(f"False Negative count = {fn_count_legacy}")
    if (tp_count_legacy + fp_count_legacy + tn_count_legacy + fn_count_legacy) != 0:
        print(f"Accuracy             = {round(100.0*(tp_count_legacy+tn_count_legacy)/(tp_count_legacy+fp_count_legacy+fn_count_legacy+tn_count_legacy),2)}")
    else:
        print(f"Accuracy             = N/A (no classified apps)")
    if (tp_count_legacy + fp_count_legacy) != 0:
        print(f"Precision            = {round(100.0*tp_count_legacy/(tp_count_legacy+fp_count_legacy),2)}")
    else:
        print(f"Precision            = N/A (no predicted positive apps)")
    if (tp_count_legacy + fn_count_legacy) != 0:
        print(f"Recall               = {round(100.0*tp_count_legacy/(tp_count_legacy+fn_count_legacy),2)}")
    else:
        print(f"Recall               = N/A (no actual positive apps)")

output_data = {
    'cpu_log': [cpu_log],
    'gpu_log': [gpu_log],
    'tp_count': [tp_count],
    'fp_count': [fp_count],
    'tn_count': [tn_count],
    'fn_count': [fn_count],
    'tp_count_legacy': [tp_count_legacy],
    'fp_count_legacy': [fp_count_legacy],
    'tn_count_legacy': [tn_count_legacy],
    'fn_count_legacy': [fn_count_legacy]
}
output_df = pd.DataFrame(output_data)
output_df.to_csv(f"{output}/summary.csv", index=False)
