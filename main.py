import base64
import logging
import logging.config

import yaml  # type: ignore
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import BaseModel, Field

from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline
from myocr.pipelines.structured_output_pipeline import StructuredOutputOCRPipeline
from myocr.utils import extract_image_type

app = Flask(__name__)
logger = logging.getLogger(__name__)


CORS(app, resources={r"/ocr": {"origins": "*"}, r"/ocr-json": {"origins": "*"}})

with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


class InvoiceItem(BaseModel):
    name: str = Field(description="发票中的项目名称")
    price: float = Field(description="项目单价")
    number: str = Field(description="项目数量")
    tax: str = Field(description="项目税额，请转为两位小数表示")

    def to_dict(self):
        return self.__dict__


class InvoiceModel(BaseModel):
    invoiceNumber: str = Field(description="发票号码，一般在发票的又上角")
    invoiceDate: str = Field(
        description="发票日期，每张发票都有一个开票日期，一般在发票的右上角，请用这种格式展示 yyyy/MM/DD"
    )
    invoiceItems: list[InvoiceItem] = Field(
        description="发票中的项目列表，这是发票中的主要内容，一般包含项目的名称，单价，数量，总价，税率，税额等，注意：这个字段是数组类型"
    )
    totalAmount: float = Field(description="发票的总金额")

    def to_dict(self):
        self.__dict__["invoiceItems"] = [item.__dict__ for item in self.invoiceItems]
        return self.__dict__


common_ocr_pipeline = CommonOCRPipeline("cuda:0")
pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)


@app.route("/ping")
def ping():
    return "pong"


def _do_ocr(pipeline):
    try:
        image_data = request.json.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        image_type, base64_data = extract_image_type(image_data)
        if not image_type:
            return jsonify({"error": "Invalid base64 image data"}), 400

        image_bytes = base64.b64decode(base64_data)
        result = pipeline(image_bytes)
        if result is None:
            return jsonify({"error": "Failed to process image, no text detected"}), 400
        return jsonify(result.to_dict())
    except Exception as e:
        logger.error("OCR error", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/ocr", methods=["POST"])
def ocr():
    return _do_ocr(common_ocr_pipeline)


@app.route("/ocr-json", methods=["POST"])
def ocr_json():
    # set default template: invoice model
    template = request.json.get("template")
    if template is None or template == "invoice":
        pipeline.set_response_format(InvoiceModel)
    else:
        return jsonify({"error": f"Not supported template: {template}"}), 400
    return _do_ocr(pipeline)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
