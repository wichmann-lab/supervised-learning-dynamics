import os

import utils.helper as h

######Colors######
CLR_MOVING_AVRG = ['#fcc5c0', '#dd3497', '#49006a']
CLR_MODELS = ['#ffc966', '#66b2b2']


######Models######
MODELS = ['resnet50', 'alexnet', 'vgg16', 'vit', 'convnext', 'efficientnet']

MODELS_DICT = {'resnet50': 'ResNet-50', 'alexnet': 'AlexNet', 'vgg16': 'VGG-16',
              'vit': 'ViT', 'convnext': 'ConvNeXt', 'efficientnet': 'EfficientNet',
              'human': 'Humans'}

MODELS_DICT_PARAM = {'resnet50': 'ResNet-50\n(25.6M)', 'alexnet': 'AlexNet\n(81.1M)', 'vgg16': 'VGG-16\n(138.4M)',
              'vit': 'ViT\n(86.6M)', 'convnext': 'ConvNeXt\n(88.6M)', 'efficientnet': 'EfficientNet\n(54.1M)',
              'human': 'Humans'}

CLASSIC_MODELS_DICT = {'resnet50': 'ResNet-50', 'alexnet': 'AlexNet', 'vgg16': 'VGG-16'}

SOTA_MODELS_DICT = {'vit': 'ViT', 'convnext': 'ConvNeXt', 'efficientnet': 'EfficientNet'}


######Paths######
CWD = os.getcwd()
PATH_DATA= h.dir_up(CWD, 1) + '/data/'
PATH_PLOTS = h.dir_up(CWD, 1) + '/plots/'

#####Training objects#####

TRAINING_OBJECTS = ['second_gen_v_86', 'second_gen_v_15', 'second_gen_q_45', 'second_gen_q_92',
                    'second_gen_v_42', 'second_gen_v_18', 'second_gen_q_70', 'second_gen_p_83',
                    'second_gen_q_83', 'second_gen_q_81', 'second_gen_v_58', 'second_gen_p_8',
                    'second_gen_p_3', 'second_gen_v_8', 'second_gen_p_4', 'second_gen_p_90',
                    'second_gen_p_18', 'second_gen_q_66']