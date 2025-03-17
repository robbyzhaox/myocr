import os

# os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = (
    os.environ.get("MYOCR_MODULE_PATH")
    or os.environ.get("MODULE_PATH")
    or os.path.expanduser("~/.MyOCR/")
)

detection_models = {
    'craft' : {
        'filename': 'craft_mlt_25k.pth',
        'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip',
        'md5sum': '2f8227d2def4037cdb3b34389dcf9ec1'
    },
    'dbnet18' : {
        'filename': 'pretrained_ic15_res18.pt',
        'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.0/pretrained_ic15_res18.zip',
        'md5sum': 'aee04f8ffe5fc5bd5abea73223800425'
    },
    'dbnet50' : {
        'filename': 'pretrained_ic15_res50.pt',
        'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.0/pretrained_ic15_res50.zip',
        'md5sum': 'a8e90144c131c2467d1eb7886c2e93a6'
    }
}