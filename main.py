import base64
import logging
import logging.config

import yaml  # type: ignore
from flask import Flask, jsonify, request
from flask_cors import CORS

from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline
from myocr.pipelines.response_format import InvoiceModel
from myocr.pipelines.structured_output_pipeline import StructuredOutputOCRPipeline
from myocr.utils import extract_image_type

app = Flask(__name__)
logger = logging.getLogger(__name__)


CORS(app, resources={r"/ocr": {"origins": "*"}, r"/ocr-json": {"origins": "*"}})

with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


@app.route("/ping")
def ping():
    return "pong"


common_ocr_pipeline = CommonOCRPipeline("cuda:0")
pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)


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
