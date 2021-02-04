from yolact_utils.functions import SavePath
from os.path import dirname, abspath, join

"""
Yolact configs variables
"""

# Yolact weights path (default: ~/weights/yolact_resnet50_cilia_133_5614_interrupt.pth)
# ATTENTION! Pattern of file name must be like: 'model name'_'epoch'_'iteration'
yolact_weights = join(dirname(abspath('main.py')), 'weights', 'yolact_resnet50_cilia_133_5614_interrupt.pth')

# Load yolact config TODO CAMBIARE IN UN METODO
yolact_model_path = SavePath.from_str(yolact_weights)
yolact_config = yolact_model_path.model_name + '_config'

# Yolact threshold
yolact_threshold = 0

# Max number of segmentations instances to get
yolact_num_max_predictions = 1