checkpoint_config = dict(           # Checkpoint hook 的配置文件。执行时请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py。
    interval=1                          # 保存的间隔是 1。
)
# yapf:disable
log_config = dict(                  # register logger hook 的配置文件。
    interval=50,                        # 打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook'),    # 用于记录训练过程的记录器(logger)。
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')  # 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'                  # 日志的级别。
load_from = None                    # 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间。
resume_from = None                  # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]           # runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次。根据 total_epochs 工作流训练 12个回合。

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
