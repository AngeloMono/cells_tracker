from os.path import dirname, abspath, join
from torch import device as torch_device, set_default_tensor_type
from torch.cuda import is_available
import torch.backends.cudnn as cudnn
from yolact import Yolact
from data import set_cfg, cfg
from yolact_utils.functions import SavePath


"""
Yolact configs variables
"""

# Yolact weights path (default: ~/weights/yolact_resnet50_cilia_133_5614_interrupt.pth)
# ATTENTION! Pattern of file name must be like: 'model name'_'epoch'_'iteration'
yolact_weights = join(dirname(abspath('main.py')), 'weights', 'yolact_resnet50_cilia_133_5614_interrupt.pth')

# Yolact threshold
yolact_threshold = 0

# Max number of segmentations instances to get
yolact_num_max_predictions = 1


def initialize_yolact_model(yolact_weights_path, cuda: bool = None):
    """
    Load and initialize the Yolact model with the weight specified in the yolact_weights_path
    :param yolact_weights_path: path of the yolact weights
    :param cuda: specifies whether the device has cuda device
    :return: the Yolact net loaded
    """
    if cuda is None:
        device = torch_device('cuda' if is_available() else 'cpu')  # Torch device used
        cuda = device.type != 'cpu'

    yolact_model_path = SavePath.from_str(yolact_weights_path)
    yolact_config = yolact_model_path.model_name + '_config'
    set_cfg(yolact_config)
    if cuda:
        cudnn.fastest = True
        set_default_tensor_type('torch.cuda.FloatTensor')
    yolact_model = Yolact()
    yolact_model.load_weights(yolact_weights_path)
    yolact_model.eval()
    if cuda:
        yolact_model = yolact_model.cuda()
    yolact_model.detect.use_fast_nms = True
    cfg.mask_proto_debug = False
    return yolact_model
