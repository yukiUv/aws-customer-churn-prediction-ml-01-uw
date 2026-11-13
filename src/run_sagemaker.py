from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import (
    SourceCode,
    Compute,
    InputData,
)
from sagemaker.core.shapes.shapes import(
    OutputDataConfig,
    StoppingCondition,
    
)
import boto3

from sagemaker.core.helper.session_helper import Session

# AWS config
ROLE_ARN = (
    "arn:aws:iam::945461161707:"
    "role/SageMakerChurnExecutionRole"
)

TRAINING_IMAGE =(
    "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/"
    "sagemaker-scikit-learn:1.9-0-cpu-py3"
)

S3_BUCKET = "aws-ml-churn-945461161707"


# Create AWS session using the churn-project profile
boto_session = boto3.Session(
    profile_name="churn-project",
    region_name="ap-southeast-2"
)

# Create a SageMaker session using the AWS session

sm_session = Session(
    boto_session=boto_session,
    default_bucket="aws-ml-churn-945461161707"
)

# config sagemaker

trainer = ModelTrainer(
    role=ROLE_ARN,
    sagemaker_session=sm_session,
    base_job_name="customer-churn",
    training_image=TRAINING_IMAGE,
    
    source_code=SourceCode(
        source_dir="src",
        entry_script="sagemaker_train.py",
    ),
    
    compute=Compute(
        instance_type="ml.m5.large",
        instance_count=1,
        volume_size_in_gb=30,
    ),
    
    output_data_config=OutputDataConfig(
        s3_output_path=f"s3://{S3_BUCKET}/sagemaker-output/"   
    ),
    
    stopping_condition=StoppingCondition(
        max_runtime_in_seconds=3600    
    ),
    
    environment={
    "PYTHONUNBUFFERED": "1",
    },
    
)

# config the training dataset
training_data = InputData(
    channel_name="train",
    data_source=f"s3://{S3_BUCKET}/data/",
    content_type="text/csv",
    
)

print("SageMaker configuration created successfully.")
print("Training image:", TRAINING_IMAGE)
print("Training instance: ml.m5.large")
print("Dataset:", training_data.data_source)

print("\nValidating configuration using dry_run=True....")

# build the training request without submitting a job
training_request = trainer._create_training_job_args(
    input_data_config=[training_data]
)

print("\nTRAINING REQUEST DETAILS")
print("------------------------")

print("Training job name:")
print(training_request.get("training_job_name"))

print("\nTraining configuration:")
print(training_request.get("resource_config"))

print("\nTraining image:")
print(
    training_request.get("algorithm_specification").training_image
)

algorithm = training_request.get("algorithm_specification")

print("\nContainer entry point:")
print(algorithm.container_entrypoint)

print("\nContainer argumnets:")
print(algorithm.container_arguments)

print("\nInput data:")
print(training_request.get("input_data_config"))

print("\nOutput configuration:")
print(training_request.get("output_data_config"))

print("\nExecution role:")
print(training_request.get("role_arn"))

print("\nRequest inspection completed.")
