import base64
import logging
import uuid
from pathlib import Path
from tempfile import gettempdir

from flask import Flask, jsonify, request
from flask_cors import CORS

from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline
from myocr.pipelines.response_format import InvoiceModel
from myocr.pipelines.structured_output_pipeline import StructuredOutputOCRPipeline

app = Flask(__name__)
logger = logging.getLogger(__name__)


CORS(app, resources={r"/ocr": {"origins": "*"}, r"/ocr-json": {"origins": "*"}})


@app.route("/ping")
def ping():
    return "pong"


common_ocr_pipeline = CommonOCRPipeline("cuda:0")
pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)


def check_temp_dir():
    """确保临时目录存在"""
    temp_dir = Path(gettempdir()) / ".temp"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def _do_ocr(pipeline):
    try:
        image_data = request.json.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        def extract_image_type(base64_data):
            if base64_data.startswith("data:image/"):
                prefix_end = base64_data.find(";base64,")
                if prefix_end != -1:
                    return (
                        base64_data[len("data:image/") : prefix_end],
                        base64_data.split(";base64,")[-1],
                    )
            return "png", base64_data

        image_type, base64_data = extract_image_type(image_data)
        if not image_type:
            return jsonify({"error": "Invalid base64 image data"}), 400

        temp_dir = check_temp_dir()
        filename = temp_dir / f"{uuid.uuid4()}.{image_type}"

        try:
            image_bytes = base64.b64decode(base64_data)
            with open(filename, "wb") as f:
                f.write(image_bytes)

            result = pipeline(str(filename))
            if result is None:
                return jsonify({"error": "Failed to process image, no text detected"}), 400

            return jsonify({"data": result.to_dict()})
        except Exception as inner_error:
            logger.error(f"Error processing image: {inner_error}", exc_info=True)
            return jsonify({"error": f"Image processing error: {str(inner_error)}"}), 500
        finally:
            # 确保清理临时文件
            if filename.exists():
                filename.unlink()
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
    check_temp_dir()
    app.run(host="0.0.0.0", port=5000, threaded=True)
