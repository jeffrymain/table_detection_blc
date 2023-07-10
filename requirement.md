# 安装命令
```
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    pip install Cython
    # 安装pycocotools
    # pip install openmim
    # mim install mmdet==2.23.0
    pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
    pip install mmdet==2.23.0
    pip install opencv-python
```

# 可能出现的问题

## 清华源

pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

## TYPEERROR: 'NUMPY.FLOAT64' OBJECT CANNOT BE INTERPRETED AS AN INTEGER

两种解决方法：

- 将`numpy`换成低版本，`pip install numpy==1.16.0`
- 将`pycocotools`库下`cocoeval.py`里的506，507行换成
    ```
    self.iouThrs = np.linspace(.5, 0.95, 10, endpoint=True)
    self.recThrs = np.linspace(.0, 1.00, 101, endpoint=True)
    self.iouThrs[8] = 0.9
    ```
