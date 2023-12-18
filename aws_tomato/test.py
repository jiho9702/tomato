from amazon_rekognition_control.amazon_control import AmazonControl

def main():

    amazon = AmazonControl()
    amazon.start_rekognition()
    # amazon.stop_rekognition()


if __name__ == "__main__":
    main()