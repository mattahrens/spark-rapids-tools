# spark-rapids-user-tools

User tools to help with the adoption, installation, execution, and tuning of RAPIDS Accelerator for Apache Spark.

The wrapper improves end-user experience within the following dimensions:
1. Qualification: Educate the CPU customer on the cost savings and acceleration potential of RAPIDS Accelerator for
   Apache Spark. The output shows a list of apps recommended for RAPIDS Accelerator for Apache Spark with estimated savings
   and speed-up.
2. Bootstrap: Provide optimized RAPIDS Accelerator for Apache Spark configs based on Dataproc GPU cluster shape. The output
   shows updated Spark config settings on Dataproc master node.
3. Tuning: Tune RAPIDS Accelerator for Apache Spark configs based on initial job run leveraging Spark event logs. The output
   shows recommended per-app RAPIDS Accelerator for Apache Spark config settings.
4. Diagnostics: Run diagnostic functions to validate the Dataproc with RAPIDS Accelerator for Apache Spark environment to
   make sure the cluster is healthy and ready for Spark jobs.


## Getting started

set python environment to version [3.8, 3.10]

1. Run the project in a virtual environment.
    ```sh
    $ python -m venv .venv
    $ source .venv/bin/activate
    ```
2. Install spark-rapids-user-tools 
    - Using released package.
      
      ```sh
      $ pip install spark-rapids-user-tools
      ```
    
3. Make sure to install gcloud SDK if you plan to run the tool wrapper.

## Using the Rapids Tool Wrapper

The wrapper provides convenient way to run Qualification/Profiling tool.
Default properties can be set in `resources/qual-conf.yaml`

- run the help command `spark_rapids_dataproc --help`

  ```bash
  NAME
      spark_rapids_dataproc - A wrapper script to run Rapids Qualification/Profiling tools on DataProc

  SYNOPSIS
      spark_rapids_dataproc <TOOL> - where tool is one of following: qualification, profiling and boostrap
      For details on the argument of each tool
      spark_rapids_dataproc <TOOL> --help

  DESCRIPTION
      Disclaimer:
        Estimates provided by the tools are based on the currently supported "SparkPlan" or
        "Executor Nodes" used in the application. It currently does not handle all the expressions
        or datatypes used.
        The pricing estimate does not take into considerations:
        1- Sustained Use discounts
        2- Cost of on-demand VMs

  ```

- run the qualification tool help cmd `spark_rapids_dataproc qualification --help`
    ```
    NAME
        spark_rapids_dataproc qualification - The Qualification tool analyzes Spark events generated from
        CPU based Spark applications to help quantify the expected acceleration and costs savings of
        migrating a Spark application or query to GPU.
    
    SYNOPSIS
        spark_rapids_dataproc qualification CLUSTER REGION <flags>
    
    DESCRIPTION
        Disclaimer:
            Estimates provided by the Qualification tool are based on the currently supported "SparkPlan" or
            "Executor Nodes" used in the application.
            It currently does not handle all the expressions or datatypes used.
            Please refer to "Understanding Execs report" section and the "Supported Operators" guide
            to check the types and expressions you are using are supported.
    
    POSITIONAL ARGUMENTS
        CLUSTER
            Type: str
            Name of the dataproc cluster
        REGION
            Type: str
            Compute region (e.g. us-central1) for the cluster.
    
    FLAGS
        --tools_jar=TOOLS_JAR
            Type: Optional[str]
            Default: None
            Path to a bundled jar including Rapids tool. The path is a local filesystem, or gstorage url.
        --eventlogs=EVENTLOGS
            Type: Optional[str]
            Default: None
            Event log filenames(comma separated) or gcloud storage directories containing event logs.
            eg: gs://<BUCKET>/eventlog1,gs://<BUCKET1>/eventlog2 If not specified, the wrapper will pull
            the default SHS directory from the cluster properties, which is equivalent to
            gs://$temp_bucket/$uuid/spark-job-history or the PHS log directory if any.
        --output_folder=OUTPUT_FOLDER
            Type: str
            Default: '.'
            Base output directory. The final output will go into a subdirectory called wrapper-output.
            It will overwrite any existing directory with the same name.
        --filter_apps=FILTER_APPS
            Type: str
            Default: 'savings'
            [NONE | recommended | savings] filtering criteria of the applications listed in the final
            STDOUT table. Note that this filter does not affect the CSV report. “NONE“ means no filter
            applied. “recommended“ lists all the apps that are either 'Recommended', or
            'Strongly Recommended'. “savings“ lists all the apps that have positive estimated GPU savings.
        --gpu_device=GPU_DEVICE
            Type: str
            Default: 'T4'
            The type of the GPU to add to the cluster. Options are [T4, V100, K80, A100, P100].
        --gpu_per_machine=GPU_PER_MACHINE
            Type: int
            Default: 2
            The number of GPU accelerators to be added to each VM image.
        --cuda=CUDA
            Type: str
            Default: '11.5'
            cuda version to be used with the GPU accelerator.
        --debug=DEBUG
            Type: bool
            Default: False
            True or False to enable verbosity to the wrapper script.
        Additional flags are accepted.
            A list of valid Qualification tool options. Note that the wrapper ignores the
            “output-directory“ flag, and it does not support multiple “spark-property“ arguments.
            For more details on Qualification tool options, please visit
            https://nvidia.github.io/spark-rapids/docs/spark-qualification-tool.html#qualification-tool-options.
    ```

- Example: Running Qualification tool passing list of google storage directories
  - Note that the wrapper lists the applications with positive recommendations.
    To list all the applications, set the argument `--filter_apps=NONE`
  - cmd
    ```
    spark_rapids_dataproc \
        qualification \
        --cluster=ahussein-jobs-test-003 \
        --region=us-central1 \
        --eventlogs=gs://mahrens-test/qualification_testing/dlrm_cpu/,gs://mahrens-test/qualification_testing/tpcds_100in1/
    ```
  - result
    ```
    Qualification tool output is saved to local disk /Users/ahussein/workspace/repos/issues/umbrella-dataproc/repos/issues/spark-rapids-tools-35-b/wrapper-output/spark_rapids_dataproc_qualification/qual-tool-output/rapids_4_spark_qualification_output
            rapids_4_spark_qualification_output/
                    ├── rapids_4_spark_qualification_output.log
                    ├── rapids_4_spark_qualification_output.csv
                    ├── rapids_4_spark_qualification_output_execs.csv
                    ├── rapids_4_spark_qualification_output_stages.csv
                    └── ui/
    - To learn more about the output details, visit https://nvidia.github.io/spark-rapids/docs/spark-qualification-tool.html#understanding-the-qualification-tool-output
    Full savings and speedups CSV report: /Users/ahussein/workspace/repos/issues/umbrella-dataproc/repos/issues/spark-rapids-tools-35-b/wrapper-output/spark_rapids_dataproc_qualification/qual-tool-output/rapids_4_dataproc_qualification_output.csv
    +----+-------------------------+---------------------+----------------------+-----------------+-----------------+---------------+-----------------+
    |    | App ID                  | App Name            | Recommendation       |   Estimated GPU |   Estimated GPU |           App |   Estimated GPU |
    |    |                         |                     |                      |         Speedup |     Duration(s) |   Duration(s) |      Savings(%) |
    |----+-------------------------+---------------------+----------------------+-----------------+-----------------+---------------+-----------------|
    |  0 | app-20200423035604-0002 | spark_data_utils.py | Strongly Recommended |            3.66 |          651.24 |       2384.32 |           64.04 |
    |  1 | app-20200423035119-0001 | spark_data_utils.py | Strongly Recommended |            3.14 |           89.61 |        281.62 |           58.11 |
    |  2 | app-20200423033538-0000 | spark_data_utils.py | Strongly Recommended |            3.12 |          300.39 |        939.21 |           57.89 |
    |  3 | app-20210509200722-0001 | Spark shell         | Strongly Recommended |            2.55 |          698.16 |       1783.65 |           48.47 |
    +----+-------------------------+---------------------+----------------------+-----------------+-----------------+---------------+-----------------+
    Report Summary:
    ------------------------------  ------
    Total applications                   4
    RAPIDS candidates                    4
    Overall estimated speedup         3.10
    Overall estimated cost savings  57.50%
    ------------------------------  ------
    To launch a GPU-accelerated cluster with RAPIDS Accelerator for Apache Spark, add the following to your cluster creation script:
            --initialization-actions=gs://goog-dataproc-initialization-actions-us-central1/gpu/install_gpu_driver.sh,gs://goog-dataproc-initialization-actions-us-central1/rapids/rapids.sh \ 
            --worker-accelerator type=nvidia-tesla-t4,count=2 \ 
            --metadata gpu-driver-provider="NVIDIA" \ 
            --metadata rapids-runtime=SPARK \ 
            --cuda-version=11.5
    ```

        
- Example: Running Qualification tool a passing list of google storage directories when cluster is running an n2 instance. N2 instances don't support GPU at the time of writing this tool and so the tool will recommend an equivalent n1 instance and run the qualification using that instance.
  - Note that the wrapper lists the applications with positive recommendations.
    To list all the applications, set the argument `--filter_apps=NONE`.
  - cmd
    ```
    spark_rapids_dataproc \
        qualification \
        --cluster=dataproc-wrapper-test \
        --region=us-central1 \
        --eventlogs=gs://mahrens-test/qualification_testing/dlrm_cpu/,gs://mahrens-test/qualification_testing/tpcds_100in1/
    ```
  - result
    ```
    Qualification tool output is saved to local disk /Users/ahussein/workspace/repos/issues/umbrella-dataproc/repos/issues/spark-rapids-tools-35-b/wrapper-output/spark_rapids_dataproc_qualification/qual-tool-output/rapids_4_spark_qualification_output
            rapids_4_spark_qualification_output/
                    ├── rapids_4_spark_qualification_output.log
                    ├── rapids_4_spark_qualification_output.csv
                    ├── rapids_4_spark_qualification_output_execs.csv
                    ├── rapids_4_spark_qualification_output_stages.csv
                    └── ui/
    - To learn more about the output details, visit https://nvidia.github.io/spark-rapids/docs/spark-qualification-tool.html#understanding-the-qualification-tool-output
    Full savings and speedups CSV report: /Users/ahussein/workspace/repos/issues/umbrella-dataproc/repos/issues/spark-rapids-tools-35-b/wrapper-output/spark_rapids_dataproc_qualification/qual-tool-output/rapids_4_dataproc_qualification_output.csv
    +----+---------------------+-------------------------+----------------------+-----------------+-----------------+---------------+-----------------+
    |    | App Name            | App ID                  | Recommendation       |   Estimated GPU |   Estimated GPU |           App |   Estimated GPU |
    |    |                     |                         |                      |         Speedup |     Duration(s) |   Duration(s) |      Savings(%) |
    |----+---------------------+-------------------------+----------------------+-----------------+-----------------+---------------+-----------------|
    |  0 | spark_data_utils.py | app-20200423035604-0002 | Strongly Recommended |            3.04 |          783.38 |       2384.32 |           27.25 |
    |  1 | spark_data_utils.py | app-20200423033538-0000 | Strongly Recommended |            2.86 |          327.36 |        939.21 |           22.82 |
    |  2 | spark_data_utils.py | app-20200423035119-0001 | Strongly Recommended |            2.69 |          104.35 |        281.62 |           17.95 |
    |  3 | Spark shell         | app-20210509200722-0001 | Recommended          |            2.25 |          789.90 |       1783.65 |            1.94 |
    +----+---------------------+-------------------------+----------------------+-----------------+-----------------+---------------+-----------------+
    Report Summary:
    ------------------------------  ------
    Total applications                   4
    RAPIDS candidates                    4
    Overall estimated acceleration    3.10
    Overall estimated cost savings  57.50%
    ------------------------------  ------
    
    To support acceleration with T4 GPUs, you will need to switch your worker node instance type to n1-highcpu-32
    To launch a GPU-accelerated cluster with RAPIDS Accelerator for Apache Spark, add the following to your cluster creation script:
            --initialization-actions=gs://goog-dataproc-initialization-actions-us-central1/gpu/install_gpu_driver.sh,gs://goog-dataproc-initialization-actions-us-central1/rapids/rapids.sh \ 
            --worker-accelerator type=nvidia-tesla-t4,count=2 \ 
            --metadata gpu-driver-provider="NVIDIA" \ 
            --metadata rapids-runtime=SPARK \ 
            --cuda-version=11.5
    ```

- run the profiling tool help cmd `spark_rapids_dataproc profiling --help`
  ```bash
  NAME
      spark_rapids_dataproc profiling - The Profiling tool analyzes both CPU or GPU generated event
      logs and generates information which can be used for debugging and profiling Apache Spark applications.

  SYNOPSIS
      spark_rapids_dataproc profiling CLUSTER REGION <flags>

  DESCRIPTION
      The output information contains the Spark version, executor details, properties, etc. It also
      uses heuristics based techniques to recommend Spark configurations for users to run Spark on RAPIDS.

  POSITIONAL ARGUMENTS
      CLUSTER
          Type: str
          Name of the dataproc cluster
      REGION
          Type: str
          Compute region (e.g. us-central1) for the cluster.

  FLAGS
      --tools_jar=TOOLS_JAR
          Type: Optional[str]
          Default: None
          Path to a bundled jar including Rapids tool. The path is a local filesystem, or gstorage url.
      --eventlogs=EVENTLOGS
          Type: Optional[str]
          Default: None
          Event log filenames(comma separated) or gcloud storage directories containing event logs.
          eg: gs://<BUCKET>/eventlog1,gs://<BUCKET1>/eventlog2 If not specified, the wrapper will pull
          the default SHS directory from the cluster properties, which is equivalent to
          gs://$temp_bucket/$uuid/spark-job-history or the PHS log directory if any.
      --output_folder=OUTPUT_FOLDER
          Type: str
          Default: '.'
          Base output directory. The final output will go into a subdirectory called wrapper-output.
          It will overwrite any existing directory with the same name.
      --debug=DEBUG
          Type: bool
          Default: False
          True or False to enable verbosity to the wrapper script.
      Additional flags are accepted.
        A list of valid Profiling tool options. Note that the wrapper ignores the following flags
        ["auto-tuner", "worker-info", "compare", "combined", "output-directory"]. For more details
        on Profiling tool options, please visit https://nvidia.github.io/spark-rapids/docs/spark-profiling-tool.html#profiling-tool-options.

  ```

- Example Running Profiling tool passing list of google storage directories
    - cmd
      ```
      spark_rapids_dataproc \
          profiling \
          --cluster=ahussein-jobs-test-003 \
          --region=us-central1 \
          --eventlogs=gs://mahrens-test/profile_testing/otherexamples/
      ```
    - result
      ```bash
      2022-09-23 13:25:17,040 INFO profiling: Preparing remote Work Env
      2022-09-23 13:25:18,242 INFO profiling: Upload Dependencies to Remote Cluster
      2022-09-23 13:25:20,163 INFO profiling: Running the tool as a spark job on dataproc
      2022-09-23 13:25:59,142 INFO profiling: Downloading the tool output
      2022-09-23 13:26:02,233 INFO profiling: Processing tool output
      Processing App app-20210507103057-0000
      Processing App app-20210413122423-0000
      Processing App app-20210507105707-0001
      Processing App app-20210422144630-0000
      Processing App app-20210609154416-0002
      ```

- run the bootstrap tool help cmd `spark_rapids_dataproc bootstrap --help`
  ```bash
  NAME
      dataproc_wrapperx.py bootstrap - Provide optimized RAPIDS Accelerator for Apache Spark configs based on Dataproc GPU cluster shape.

  SYNOPSIS
      dataproc_wrapperx.py bootstrap CLUSTER REGION <flags>

  DESCRIPTION
      Provide optimized RAPIDS Accelerator for Apache Spark configs based on Dataproc GPU cluster shape.

  POSITIONAL ARGUMENTS
      CLUSTER
          Type: str
          Name of the dataproc cluster
      REGION
          Type: str
          Compute region (e.g. us-central1) for the cluster.

  FLAGS
      --dry_run=DRY_RUN
          Type: bool
          Default: False
          True or False to update the Spark config settings on Dataproc master node.
      --debug=DEBUG
          Type: bool
          Default: False
          True or False to enable verbosity to the wrapper script.
  ```

- Example Running bootstrap tool
    - cmd
      ```
      spark_rapids_dataproc \
          bootstrap \
          --cluster=ahussein-jobs-test-003 \
          --region=us-central1
      ```
    - result
      ```bash
      ##### BEGIN : RAPIDS bootstrap settings for ahussein-jobs-test-003
      spark.executor.cores=8
      spark.executor.memory=16384m
      spark.executor.memoryOverhead=5734m
      spark.rapids.sql.concurrentGpuTasks=2
      spark.rapids.memory.pinnedPool.size=4096m
      spark.sql.files.maxPartitionBytes=512m
      spark.task.resource.gpu.amount=0.125
      ##### END : RAPIDS bootstrap settings for ahussein-jobs-test-003
      ```

## Running Diagnostic Tool

- Run the diagnostic tool help cmd `spark_rapids_dataproc diagnostic --help`

```text
NAME
    spark_rapids_dataproc diagnostic - Run diagnostic on local environment or remote Dataproc cluster,
    such as check installed NVIDIA driver, CUDA toolkit, RAPIDS Accelerator for Apache Spark jar etc.

SYNOPSIS
    spark_rapids_dataproc diagnostic CLUSTER REGION <flags>

DESCRIPTION
    Run diagnostic on local environment or remote Dataproc cluster, such as check installed NVIDIA driver,
    CUDA toolkit, RAPIDS Accelerator for Apache Spark jar etc.

POSITIONAL ARGUMENTS
    CLUSTER
        Type: str
        Name of the Dataproc cluster
    REGION
        Type: str
        Region of Dataproc cluster (e.g. us-central1)

FLAGS
    --func=FUNC
        Type: str
        Default: 'all'
        Diagnostic function to run. Available functions: 'nv_driver': dump NVIDIA driver info via command
        `nvidia-smi`, 'cuda_version': check if CUDA toolkit major version >= 11.0, 'rapids_jar': check if
        only single RAPIDS Accelerator for Apache Spark jar is installed and verify its signature, 'deprecated_jar': check if deprecated
        (cudf) jar is installed. I.e. should no cudf jar starting with RAPIDS Accelerator for Apache Spark 22.08, 'spark': run a
        Hello-world Spark Application on CPU and GPU, 'perf': performance test for a Spark job between CPU and
        GPU, 'spark_job': run a Hello-world Spark Application on CPU and GPU via Dataproc job interface, 'perf_job':
        performance test for a Spark job between CPU and GPU via Dataproc job interface
    --debug=DEBUG Type: bool
        Default: False
        True or False to enable verbosity

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

- Example running diagnostic tool

    - cmd

      ```bash
      spark_rapids_dataproc \
          diagnostic \
          --cluster=alex-demt \
          --region=us-central1 \
          nv_driver
      ```

    - result
      ```text
      *** Running diagnostic function "nv_driver" ***
      Warning: Permanently added 'compute.3346163243442954535' (ECDSA) to the list of known hosts.
      Wed Oct 19 02:32:36 2022
      +-----------------------------------------------------------------------------+
      | NVIDIA-SMI 460.106.00   Driver Version: 460.106.00   CUDA Version: 11.2     |
      |-------------------------------+----------------------+----------------------+
      | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
      | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
      |                               |                      |               MIG M. |
      |===============================+======================+======================|
      |   0  Tesla T4            On   | 00000000:00:04.0 Off |                    0 |
      | N/A   63C    P8    11W /  70W |      0MiB / 15109MiB |      0%      Default |
      |                               |                      |                  N/A |
      +-------------------------------+----------------------+----------------------+

      +-----------------------------------------------------------------------------+
      | Processes:                                                                  |
      |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
      |        ID   ID                                                   Usage      |
      |=============================================================================|
      |  No running processes found                                                 |
      +-----------------------------------------------------------------------------+
      Connection to 34.171.155.172 closed.
      *** Check "nv_driver": PASS ***
      *** Running diagnostic function "nv_driver" ***
      Warning: Permanently added 'compute.5880729710893392167' (ECDSA) to the list of known hosts.
      Wed Oct 19 02:32:42 2022
      +-----------------------------------------------------------------------------+
      | NVIDIA-SMI 460.106.00   Driver Version: 460.106.00   CUDA Version: 11.2     |
      |-------------------------------+----------------------+----------------------+
      | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
      | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
      |                               |                      |               MIG M. |
      |===============================+======================+======================|
      |   0  Tesla T4            On   | 00000000:00:04.0 Off |                    0 |
      | N/A   61C    P8    10W /  70W |      0MiB / 15109MiB |      0%      Default |
      |                               |                      |                  N/A |
      +-------------------------------+----------------------+----------------------+

      +-----------------------------------------------------------------------------+
      | Processes:                                                                  |
      |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
      |        ID   ID                                                   Usage      |
      |=============================================================================|
      |  No running processes found                                                 |
      +-----------------------------------------------------------------------------+
      Connection to 34.70.29.158 closed.
      *** Check "nv_driver": PASS ***
      ```

## Changelog

### [22.10.2] - 10-28-2022
   
- Support to handle tools jar arguments in the user tools wrapper
 
### [22.10.1] - 10-26-2022
  
- Initialize this project