# from myocr.processors import OcrDetectionProcessor, OcrRecognizationProcessor


def test_hello():
    print("Hello, World!")


# def test_processor():
#     image = "/home/robby/code/myocr/tests/123.png"
#     p = OcrDetectionProcessor()
#     img, origin_shape = p.dbnet.load_image(image, detection_size=500)
#     horizontal_list, free_list = p.process(image)
#     img = img.numpy().squeeze()
#     img = np.transpose(img, (2, 1, 0))
#     print(f"shape is {img.shape}")

#     print(f"shape is {len(img.shape)} {img.shape[2]}")
#     p2 = OcrRecognizationProcessor(horizontal_list, free_list)
#     p2.process(img)
