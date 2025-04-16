from pydantic import BaseModel, Field


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
