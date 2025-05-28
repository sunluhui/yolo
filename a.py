import openxlab
openxlab.login(ak='pbyzd3eawggmon03lrwv', sk='nmoqn3gg92ldjbodnw2nnmywnmz1lrew6yawpezx') # 进行登录，输入对应的AK/SK，可在个人中心查看AK/SK


from openxlab.dataset import get
get(dataset_repo='OpenDataLab/SeaPerson', target_path='/home/a10/ZH/ultralytics-main/datasets/') # 数据集下载

