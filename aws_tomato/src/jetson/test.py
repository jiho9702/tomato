#Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-custom-labels-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import io
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import cv2
import numpy as np
import pyrealsense2 as rs
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from filter.img_filter import homomorphic_filter, histogram_normalize


def real_time(rekognition_client, model):

    cap = cv2.VideoCapture(0)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()

        if not frame:
                continue

        frame = np.asanyarray(frame.get_data())

        frame = homomorphic_filter(frame)
        # frame = histogram_normalize(frame)

        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        response = rekognition_client.detect_custom_labels(
            Image = {'Bytes' : img_bytes},
            MinConfidence = 60,
            ProjectVersionArn = model
        )

        imgWidth = frame.shape[1]
        imgHeight = frame.shape[0]
        frame_pil = Image.fromarray(np.uint8(frame))
        draw = ImageDraw.Draw(frame_pil)

        point_data = []


        print("Detected labels:")
        
        for customLabel in response['CustomLabels']:
            cnt = 0
            # print('Label ' + str(customLabel['Name']))
            # print('Confidence ' + str(customLabel)['Confidence'])

            if 'Geometry' in customLabel:
                box = customLabel['Geometry']['BoundingBox']
                left = imgWidth * box['Left']
                top = imgHeight * box['Top']
                width = imgWidth * box['Width']
                height = imgHeight * box['Height']

                center_point_x = left + width/2
                center_point_y = top + height/2

                # depth_data = depth(center_point_x, center_point_y)
                point_depth = {left, top, width, height, center_point_x, center_point_y}

                # print(point_depth)
                point_data.append(point_depth)    

                # fnt = ImageFont.truetype('Library/Fonts/Arial.ttf', 50)
                fnt = ImageFont.load_default()
                draw.text((left, top), customLabel['Name'], fill = '#00d400', font=fnt)

                points = (
                     (left, top),
                     (left + width, top),
                     (left + width, top + height),
                     (left, top + height),
                     (left, top)
                )
                draw.line(points, fill='#00d400', width=5)

        
        frame = np.array(frame_pil)
        cv2.imshow("cam", frame)
        print(len(point_data))

        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
        
    cap.release()
    cv2.destroyAllWindows()

        # image.show()

def main():
    aws_access_key_id = "AKIARQUWVSEJM4LGR2II"
    aws_secret_access_key = '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'
    region_name = 'ap-northeast-2'
    model='arn:aws:rekognition:ap-northeast-2:104468943122:project/CherryTomato/version/CherryTomato.2023-12-01T16.17.45/1701418665992'
    min_confidence=95

    rekognition_client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
    real_time(rekognition_client, model)


if __name__ == "__main__":
    main()