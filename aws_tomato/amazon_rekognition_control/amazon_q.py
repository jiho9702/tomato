#Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-custom-labels-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import time


def stop_model(model_arn):

    aws_access_key_id = "AKIARQUWVSEJM4LGR2II"
    aws_secret_access_key = 'syFzMbtDVTg8CPi7zqzt4dI/jcuMPz6Ihmt/heMT'
    region_name = 'ap-northeast-2'

    client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)

    print('Stopping model:' + model_arn)

    #Stop the model
    try:
        response=client.stop_project_version(ProjectVersionArn=model_arn)
        status=response['Status']
        print ('Status: ' + status)
    except Exception as e:  
        print(e)  

    print('Done...')
    
def main():
    
    model_arn='arn:aws:rekognition:ap-northeast-2:104468943122:project/CherryTomato/version/CherryTomato.2023-12-01T16.17.45/1701418665992'
    stop_model(model_arn)

if __name__ == "__main__":
    main() 