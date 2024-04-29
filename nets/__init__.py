def get_network(network_name):
    # network_name=network_name.lower()
    if network_name=='Wpdnet':
        from .Wpdnet import Wpdnet
        return Wpdnet
    elif network_name=='FCN':
        from .FCN import FCN
        return FCN
    elif network_name=='VGG16':
        from .VGG16 import VGG16
        return VGG16
    elif network_name=='Resnet18':
        from .Resnet18 import Resnet18
        return Resnet18
    elif network_name=='Resnet34':
        from .Resnet34 import Resnet34
        return Resnet34
    elif network_name=='MobileNet':
        from .MobileNet_torch import MobileNet
        return MobileNet
    elif network_name=='MobileNetV2':
        from .MobileNetV2 import MobileNetV2
        return MobileNetV2
    elif network_name=='AdvanceMobileNetV2':
        from .AdvanceMobileNetV2 import AdvanceMobileNetV2
        return AdvanceMobileNetV2
    elif network_name=='ggcnn':
        from .ggcnn import GGCNN2
        return GGCNN2
    else:
        raise NotImplementedError("Network {} is not implemented ".format(network_name))