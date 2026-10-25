
class Config:
    def __init__(self):
        # calibration settings
        self.calib_size = 32
        self.optim_size = 1024 # 原版
        # self.optim_size = 32
        self.calib_batch_size = 32 # 一个批次就校准完了
        self.optim_batch_size = 128
        self.w_bit = 4
        self.a_bit = 4
        # self.a_bit = 32
        self.s_bit = 4 # softmax的量化
        # self.s_bit = 32 # softmax的量化
        self.qconv_a_bit = 8 # 卷积层激活的独立量化比特数（某些架构如Swin有卷积模块）
        # self.qconv_a_bit = 32 # 卷积层激活的独立量化比特数（某些架构如Swin有卷积模块）
        self.qhead_a_bit = 4 # 分类头（Head）激活的量化比特数
        # self.qhead_a_bit = 32 # 分类头（Head）激活的量化比特数
        self.matmul_head_channel_wise = True # 多头注意力中的矩阵乘法运算进行通道级量化？
        self.post_softmax_quantizer = 'adalog' # 剩余可选的量化器是log量化器、adaround和均匀量化器
        self.post_gelu_quantizer = 'adalog'
        # search settings
        self.eq_n = 128 # 论文中的参数，搜索128个参数组合？
        self.search_round = 3 # 原版
        # self.search_round = 6
        self.fpcs = True # 快速参数搜索
        self.steps = 6 # 搜索6轮，原版
        # self.steps = 12 # 搜索6轮
        # optimization settings
        self.keep_gpu = True # 将校准数据保存在GPU上
        self.train_act = True
        self.optim_mode = 'both' # 自定义的重建优化模式：仅优化激活值
