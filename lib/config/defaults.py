from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()
_C.DATASETS.USE_ONEHOT = True
_C.DATASETS.USE_SEG = True
_C.DATASETS.USE_ATT = True
_C.DATASETS.BIN_SEG = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.IMS_PER_ID = 4
_C.DATALOADER.EN_SAMPLER = True


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 224
_C.INPUT.WIDTH = 224
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.CROP_SIZE = 32
_C.INPUT.DOWNSAMPLE_RATIO = 1 / 8
_C.INPUT.PADDING = 10
_C.INPUT.USE_AUG = False


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.VISUAL_MODEL = "resnet50"
_C.MODEL.TEXTUAL_MODEL = "bilstm"
_C.MODEL.NUM_CLASSES = 11003
_C.MODEL.NUM_PARTS = 5
_C.MODEL.FREEZE = False
_C.MODEL.WEIGHT = "imagenet"
_C.MODEL.WHOLE = False
_C.MODEL.INFERENCE_MODE = "common"


# -----------------------------------------------------------------------------
# MoCo
# -----------------------------------------------------------------------------
_C.MODEL.MOCO = CN()
_C.MODEL.MOCO.K = 1024
_C.MODEL.MOCO.M = 0.999
_C.MODEL.MOCO.FC = True


# -----------------------------------------------------------------------------
# LSTM
# -----------------------------------------------------------------------------
_C.MODEL.LSTM = CN()
_C.MODEL.LSTM.ONEHOT = True
_C.MODEL.LSTM.EMBEDDING_SIZE = 512
_C.MODEL.LSTM.NUM_UNITS = 512
_C.MODEL.LSTM.VOCABULARY_SIZE = 12000
_C.MODEL.LSTM.DROPOUT_KEEP_PROB = 0.7
_C.MODEL.LSTM.MAX_LENGTH = 100


# -----------------------------------------------------------------------------
# TextCNN
# -----------------------------------------------------------------------------
_C.MODEL.TEXT_CNN = CN()
_C.MODEL.TEXT_CNN.EMBEDDING_SIZE = 512
_C.MODEL.TEXT_CNN.FILTER_SIZE = [3, 5, 7, 9]
_C.MODEL.TEXT_CNN.NUM_FILTERS = 256
_C.MODEL.TEXT_CNN.VOCABULARY_SIZE = 12000
_C.MODEL.TEXT_CNN.DROPOUT = 0.5


# -----------------------------------------------------------------------------
# GRU
# -----------------------------------------------------------------------------
_C.MODEL.GRU = CN()
_C.MODEL.GRU.ONEHOT = "yes"
_C.MODEL.GRU.EMBEDDING_SIZE = 512
_C.MODEL.GRU.NUM_UNITS = 512
_C.MODEL.GRU.VOCABULARY_SIZE = 12000
_C.MODEL.GRU.DROPOUT_KEEP_PROB = 0.7
_C.MODEL.GRU.MAX_LENGTH = 100
_C.MODEL.GRU.NUM_LAYER = 1
_C.MODEL.GRU.GET_MASK_LABEL = False
_C.MODEL.GRU.CUT_MIX = False
_C.MODEL.GRU.RANDOM_DELETE = False
_C.MODEL.GRU.CUT_NEG = False

# -----------------------------------------------------------------------------
# BERT
# -----------------------------------------------------------------------------
_C.MODEL.BERT = CN()
_C.MODEL.BERT.POOL = True


# -----------------------------------------------------------------------------
# Resnet
# -----------------------------------------------------------------------------
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.RES5_STRIDE = 2
_C.MODEL.RESNET.RES5_DILATION = 1
_C.MODEL.RESNET.ATTN_POOL = False
_C.MODEL.RESNET.IBNA = False
_C.MODEL.RESNET.PATCH_MIX = False
_C.MODEL.RESNET.PRETRAINED = None


# -----------------------------------------------------------------------------
# Vision Transformer
# -----------------------------------------------------------------------------
_C.MODEL.VIT = CN()
_C.MODEL.VIT.MODE = "first"
_C.MODEL.VIT.CKPT = None


# -----------------------------------------------------------------------------
# Hybrid Vision Transformer
# -----------------------------------------------------------------------------
_C.MODEL.HBVIT = CN()
_C.MODEL.HBVIT.BACKBONE = "resnet50"
_C.MODEL.HBVIT.EMBED_DIM = 1024
_C.MODEL.HBVIT.DEPTH = 4
_C.MODEL.HBVIT.NUM_HEADS = 8
_C.MODEL.HBVIT.PATCH_SIZE = 1


# -----------------------------------------------------------------------------
# Hybrid GRU
# -----------------------------------------------------------------------------
_C.MODEL.HBGRU = CN()
_C.MODEL.HBGRU.EMBED_DIM = 1024
_C.MODEL.HBGRU.DEPTH = 4
_C.MODEL.HBGRU.NUM_HEADS = 8
_C.MODEL.HBGRU.FF_DIM = 4096


# -----------------------------------------------------------------------------
# Textual Transformer Encoder
# -----------------------------------------------------------------------------
_C.MODEL.TRANS_ENCODER = CN()
_C.MODEL.TRANS_ENCODER.EMBED_DIM = 512
_C.MODEL.TRANS_ENCODER.DEPTH = 4
_C.MODEL.TRANS_ENCODER.NUM_HEADS = 8
_C.MODEL.TRANS_ENCODER.FF_DIM = 1024
_C.MODEL.TRANS_ENCODER.VOCABULARY_SIZE = 12000
_C.MODEL.TRANS_ENCODER.ONEHOT = "yes"
_C.MODEL.TRANS_ENCODER.LEARN_PS = False
_C.MODEL.TRANS_ENCODER.DROPOUT = 0.1


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
_C.MODEL.EMBEDDING = CN()
_C.MODEL.EMBEDDING.EMBED_HEAD = "simple"
_C.MODEL.EMBEDDING.FEATURE_SIZE = 512
_C.MODEL.EMBEDDING.DROPOUT_PROB = 0.3
_C.MODEL.EMBEDDING.BNNECK = False
_C.MODEL.EMBEDDING.CMPC = False
_C.MODEL.EMBEDDING.CMPM = False
_C.MODEL.EMBEDDING.MIXTURE = False
_C.MODEL.EMBEDDING.K_RECIPROCAL = True
_C.MODEL.EMBEDDING.SHARED_LAYER = False
_C.MODEL.EMBEDDING.EPSILON = 0.0
_C.MODEL.EMBEDDING.TASK = ["CMR"]
_C.MODEL.EMBEDDING.LEARN_SCALE = False

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.EVALUATE_PERIOD = 1

_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.WEIGHT_DECAY = 0.00004
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999
_C.SOLVER.SGD_MOMENTUM = 0.9

_C.SOLVER.LRSCHEDULER = "step"

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.PART_STRATEGRY = False

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500,)

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.0001


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 16


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #
# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"
# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
