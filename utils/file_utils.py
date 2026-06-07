import pickle
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path
    
def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)

def add_colorbar(img, cmap, norm):
    """在图像下方添加颜色条"""
    # 创建一个包含颜色条的图
    fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')
    fig.subplots_adjust(bottom=0.5)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cbar.set_label('Attention Score')

    # 将颜色条图转换为PIL Image并调整大小
    fig.canvas.draw()  # 必须先绘制图像
    colorbar = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    colorbar = colorbar.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    colorbar = Image.fromarray(colorbar)

    # 关闭plt图以释放资源
    plt.close(fig)

    # 获取原图像的尺寸
    img_w, img_h = img.size

    # 将颜色条图像缩放到与原图同宽
    colorbar = colorbar.resize((img_w, int(img_h * 0.1)))  # 调整高度为原图的5%

    # 将颜色条图像合并到原图下方
    combined = Image.new('RGB', (img_w, img_h + colorbar.size[1]))
    combined.paste(img, (0, 0))
    combined.paste(colorbar, (0, img_h))

    return combined


def _save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def _load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file