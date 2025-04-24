## 🔍 Introduction
MyOCR is a development kit for engineers to easiy train and assemble their models to predictors and pipelines for their OCR business.


## 📣 Updates
- **🔥2025.04.24 release MyOCR alpha version**:
    - Release image detection, class, recognition models
    - All components can work together


## 📖 What Can MyOCR Do?
- **🚀 Build Your Own OCR Solutions**
We can quickly build our own OCR solutions like building blocks based on the components.

- **🚀 Easily integrate the Model and Train**
With the moduler design, we can easily integrate our custom model to MyOCR to extend the components.

- **🚀 Support Multiple Ways of Inference**
MyOCR can be used as python package to integrated to your projects, and we also support inference via Rest API.

## 📝 Qucik Start
!!! tip
    MyOCR work good on CPU and GPU, we recommend to use GPU to train and inference.

### Requirements
- Python 3.11+
- CUDA 12.6+

### Installation Method

```bash
# Clone the code from GitHub
git clone https://github.com/robbyzhaox/myocr.git
cd myocr

# Install dependencies
pip install -e .

# Development environment installation
pip install -e ".[dev]"

# Download pre-trained models
mkdir -p ~/.MyOCR/models/
curl -fsSL "https://drive.google.com/file/d/1b5I8Do4ODU9xE_dinDGZMraq4GDgHPH9/view?usp=drive_link" -o ~/.MyOCR/models/dbnet++.onnx
curl -fsSL "https://drive.google.com/file/d/1MSF7ArwmRjM4anDiMnqhlzj1GE_J7gnX/view?usp=drive_link" -o ~/.MyOCR/models/rec.onnx
curl -fsSL "https://drive.google.com/file/d/1TCu3vAXNVmPBY2KtoEBTGOE6tpma0puX/view?usp=drive_link" -o ~/.MyOCR/models/cls.onnx
```

### Local Inference
!!! example "Basic OCR Recognition"
    === "Common OCR"
        ```python
        from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline

        # Initialize OCR pipeline (using GPU)
        pipeline = CommonOCRPipeline("cuda:0")  # Use "cpu" for CPU mode

        # Perform OCR recognition on an image
        result = pipeline("path/to/your/image.jpg")
        print(result)
        ```
        ??? question "Click to See Output"
            === "Output Json"
                ```bash
                [(text=贸易战, confidence=0.0004101736412849277, bounding_box=(left=455, bottom=43, right=537, top=4, angle=(0, np.float32(0.99804187)), score=0.96864689904632))
                , (text=Text, confidence=0.0004079231293871999, bounding_box=(left=308, bottom=49, right=363, top=13, angle=(0, np.float32(0.6283184)), score=0.8928107453717127))
                , (text=文库, confidence=0.0004101540253031999, bounding_box=(left=14, bottom=93, right=79, top=49, angle=(180, np.float32(0.92576)), score=0.8890029720334343))
                , (text=图片, confidence=0.0004101961385458708, bounding_box=(left=216, bottom=94, right=278, top=49, angle=(180, np.float32(0.99998415)), score=0.857502628267903))
                , (text=地图, confidence=0.0004101953818462789, bounding_box=(left=516, bottom=94, right=576, top=49, angle=(180, np.float32(0.9999981)), score=0.872837122885572))
                , (text=贴吧, confidence=0.0004101790254935622, bounding_box=(left=415, bottom=94, right=480, top=50, angle=(180, np.float32(0.99998367)), score=0.8831927942269014))
                , (text=网盘, confidence=0.00041003606747835875, bounding_box=(left=115, bottom=96, right=179, top=52, angle=(180, np.float32(0.815267)), score=0.9080475783482278))
                , (text=视频, confidence=0.00041019057971425354, bounding_box=(left=315, bottom=96, right=379, top=52, angle=(180, np.float32(0.99999964)), score=0.9084148499800161))
                , (text=新闻, confidence=0.0004098160716239363, bounding_box=(left=751, bottom=93, right=800, top=53, angle=(180, np.float32(0.9999683)), score=0.9706179747978847))
                , (text=hao123, confidence=0.0004040453059133142, bounding_box=(left=614, bottom=89, right=714, top=54, angle=(180, np.float32(1.0)), score=0.860578941761776))
                ]
                ```
            === "Output Image"
                <p><img src="assets/images/common-ocr-output.png"></p>

    === "Structured OCR Output (Example: Invoice Information Extraction)"
        ```python
        from pydantic import BaseModel, Field
        from myocr.pipelines.structured_output_pipeline import StructuredOutputOCRPipeline

        # Define output data model
        class InvoiceItem(BaseModel):
            name: str = Field(description="Item name in the invoice")
            price: float = Field(description="Item unit price")
            number: str = Field(description="Item quantity")
            tax: str = Field(description="Item tax amount")

        class InvoiceModel(BaseModel):
            invoiceNumber: str = Field(description="Invoice number")
            invoiceDate: str = Field(description="Invoice date")
            invoiceItems: list[InvoiceItem] = Field(description="List of items in the invoice")
            totalAmount: float = Field(description="Total amount of the invoice")
            
            def to_dict(self):
                self.__dict__["invoiceItems"] = [item.__dict__ for item in self.invoiceItems]
                return self.__dict__

        # Initialize structured OCR pipeline
        pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)

        # Process image and get structured data
        result = pipeline("path/to/invoice.jpg")
        print(result.to_dict())
        ```
        ??? question "Click to See Output"
            === "Output Json"
                ```bash
                {"data":{"invoiceDate":"2018年08月15日","invoiceItems":[{"name":"飞利浦BDL6530QT (智能会议电子白板会议平板触摸一体)","number":"One unit","price":11206.03,"tax":"When it comes to the tax information, the exact rate and amount should be considered. This entry's tax is listed as 1792.97 at a 16% rate in the given invoice text."}],"invoiceNumber":"21572","totalAmount":12999.0}}
                ```
            === "Output Image"
                <p><img src="assets/images/ocr-structured-output.png"></p>

### Rest API
The framework provides a simple Flask API service that can be called via HTTP interface:

```bash
# Start the service
python main.py
```

API endpoints:
- `GET /ping`: Check if the service is running properly
- `POST /ocr`: Basic OCR recognition
- `POST /ocr-json`: Structured OCR output

We also have a UI for these endpoints, please refer to [text](https://github.com/robbyzhaox/doc-insight-ui)

## 💬 Discussion
We encourage to discuss and imporve this project to help more people. Github issues, discussions and the comments under this doc site are all available ways to discuss, try to use simple and clear language to ask questions, describe ideas, and seek help.

## 📄 License
<a href="https://github.com/robbyzhaox/myocr/blob/main/LICENSE">Apache 2.0 license</a>


