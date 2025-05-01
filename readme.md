all in a python=3.10 enviroment for ros2 humble or python=3.8 for ros2 foxy

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install -U openmim
(maybe could be necessary to update path, however youll see an stdout if necessary)

pip install mmengine

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

mim install mmdet

mim install "mmdet3d>=1.1.0"

(maybe you should downgrade numpy to <2.0)
