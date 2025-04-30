from torch import nn

from ..utils import build_component

__all__ = ["BaseModel"]


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_transform = False
        self.use_backbone = False
        self.use_neck = False
        self.use_head = False

        in_channels = config.get("in_channels", 3)
        # model_type = config["model_type"]
        if config.get("Transform") is not None:
            config["Transform"]["in_channels"] = in_channels
            self.transform = build_component("Transform", config["Transform"])
            self.use_transform = True
            in_channels = self.transform.out_channels  # type: ignore

        # build backbone
        if config.get("Backbone") is not None:
            config["Backbone"]["in_channels"] = in_channels
            self.backbone = build_component("Backbone", config["Backbone"])
            self.use_backbone = True
            in_channels = self.backbone.out_channels  # type: ignore

        # build neck
        if config.get("Neck") is not None:
            config["Neck"]["in_channels"] = in_channels
            self.neck = build_component("Neck", config["Neck"])
            self.use_neck = True
            in_channels = self.neck.out_channels  # type: ignore

        # # build head
        if config.get("Head") is not None:
            config["Head"]["in_channels"] = in_channels
            self.head = build_component("Head", config["Head"])
            self.use_head = True

        self.return_all_feats = config.get("return_all_feats", False)
        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, data=None):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"
        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"
        if self.use_head:
            x = self.head(x)
            # for multi head, save ctc neck out for udml
            # if "ctc_neck" in x.keys():
            #     y["neck_out"] = x["ctc_neck"]
            y["head_out"] = x
            final_name = "head_out"
        if self.return_all_feats:
            if self.training:
                return y
            else:
                return {final_name: x}
        else:
            return x
