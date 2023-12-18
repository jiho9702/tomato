#Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-custom-labels-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3

def start_model(project_arn, model_arn, version_name, min_inference_units):

    aws_access_key_id = "AKIARQUWVSEJM4LGR2II"
    aws_secret_access_key = 'syFzMbtDVTg8CPi7zqzt4dI/jcuMPz6Ihmt/heMT'
    region_name = 'ap-northeast-2'

    client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)

    try:
        # Start the model
        print('Starting model: ' + model_arn)
        response=client.start_project_version(ProjectVersionArn=model_arn, MinInferenceUnits=min_inference_units)
        # Wait for the model to be in the running state
        project_version_running_waiter = client.get_waiter('project_version_running')
        project_version_running_waiter.wait(ProjectArn=project_arn, VersionNames=[version_name])

        #Get the running status
        describe_response=client.describe_project_versions(ProjectArn=project_arn,
            VersionNames=[version_name])
        for model in describe_response['ProjectVersionDescriptions']:
            print("Status: " + model['Status'])
            print("Message: " + model['StatusMessage']) 
    except Exception as e:
        print(e)
        
    print('Done...')
    
def main():
    project_arn='arn:aws:rekognition:ap-northeast-2:104468943122:project/CherryTomato/1701343293786'
    model_arn='arn:aws:rekognition:ap-northeast-2:104468943122:project/CherryTomato/version/CherryTomato.2023-12-01T16.17.45/1701418665992'
    min_inference_units=1 
    version_name='CherryTomato.2023-12-01T16.17.45'
    start_model(project_arn, model_arn, version_name, min_inference_units)

if __name__ == "__main__":
    main()