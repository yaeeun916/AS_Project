import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Densenet121(BaseModel):
    def __init__(self, fc_num, dropout):
        super().__init__()
        self.pretrained = models.densenet121(pretrained=True)
        set_parameter_requires_grad(self.pretrained, feature_extracting=True)

        # modify top layer
        assert fc_num in [1, 2]
        nodes0 = self.pretrained.classifier.in_features
        nodes1 = nodes0//2
        nodes2 = nodes1//2

        if (fc_num == 1):
            self.pretrained.classifier = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, 1)
            )
        else:
            self.pretrained.classifier = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, nodes2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes2, 1)
            )

    def forward(self, x):
        x = self.pretrained(x)
        return x


class Densenet169(BaseModel):
    def __init__(self, fc_num, dropout):
        super().__init__()
        self.pretrained = models.densenet169(pretrained=True)
        set_parameter_requires_grad(self.pretrained, feature_extracting=True)

        # modify top layer
        assert fc_num in [1, 2]
        nodes0 = self.pretrained.classifier.in_features
        nodes1 = nodes0//2
        nodes2 = nodes1//2

        if (fc_num == 1):
            self.pretrained.classifier = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, 1)
            )
        else:
            self.pretrained.classifier = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, nodes2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes2, 1)
            )

    def forward(self, x):
        x = self.pretrained(x)
        return x


class Densenet201(BaseModel):
    def __init__(self, fc_num, dropout):
        super().__init__()
        self.pretrained = models.densenet201(pretrained=True)
        set_parameter_requires_grad(self.pretrained, feature_extracting=True)

        # modify top layer
        assert fc_num in [1, 2]
        nodes0 = self.pretrained.classifier.in_features
        nodes1 = nodes0//2
        nodes2 = nodes1//2

        if (fc_num == 1):
            self.pretrained.classifier = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, 1)
            )
        else:
            self.pretrained.classifier = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, nodes2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes2, 1)
            )

    def forward(self, x):
        x = self.pretrained(x)
        return x

class Resnet50(BaseModel):
    def __init__(self, fc_num, dropout):
        super().__init__()
        self.pretrained = models.resnet50(pretrained=True)
        set_parameter_requires_grad(self.pretrained, feature_extracting=True)

        # modify top layer
        assert fc_num in [1, 2]
        nodes0 = self.pretrained.fc.in_features
        nodes1 = nodes0//2
        nodes2 = nodes1//2

        if (fc_num == 1):
            self.pretrained.fc = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, 1)
            )
        else:
            self.pretrained.fc = nn.Sequential(
                nn.Linear(nodes0, nodes1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes1, nodes2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nodes2, 1)
            )

    def forward(self, x):
        x = self.pretrained(x)
        return x

# vgg19 with batch norm
class VGG19(BaseModel):
    def __init__(self, fc_num, dropout):
        super().__init__()
        self.pretrained = models.vgg19_bn(pretrained=True)
        set_parameter_requires_grad(self.pretrained, feature_extracting=True)

        # modify top layer
        assert fc_num in [1, 2]
        nodes0 = self.pretrained.classifier[6].in_features
        nodes1 = nodes0//2
        nodes2 = nodes1//2

        if (fc_num == 1):
            self.pretrained.classifier[6] = nn.Linear(nodes0, nodes1)
            self.pretrained.classifier.add_module('7', nn.ReLU())
            self.pretrained.classifier.add_module('8', nn.Dropout(dropout))
            self.pretrained.classifier.add_module('9', nn.Linear(nodes1, 1))

        else:
            self.pretrained.classifier[6] = nn.Linear(nodes0, nodes1)
            self.pretrained.classifier.add_module('7', nn.ReLU())
            self.pretrained.classifier.add_module('9', nn.Dropout(dropout))
            self.pretrained.classifier.add_module('10', nn.Linear(nodes1, nodes2))
            self.pretrained.classifier.add_module('11', nn.ReLU())
            self.pretrained.classifier.add_module('12', nn.Dropout(dropout))
            self.pretrained.classifier.add_module('13', nn.Linear(nodes2, 1))

    def forward(self, x):
        x = self.pretrained(x)
        return x

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

