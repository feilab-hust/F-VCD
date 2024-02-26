import numpy as np
import imageio
import PIL.Image as pilimg
import tensorlayer as tl
# import cv2


__all__ = [
    'get_img3d_fn',
    'rearrange3d_fn',
    'get_and_rearrange3d',
    'get_img2d_fn',
    'get_lf_extra',
    'get_2d_lf',
    'lf_extract_fn',
    'write3d',
    'normalize_percentile',
    'normalize',
    'z_score',
    'normal_clip',
    'normalize_percentile_z_score',
    'min_max',
    'normalize_constant',
    'fft',
    'save_configs',
    'binary_normal',
    'spectrum2im',
]

def save_configs(save_folder):
    import os
    from config_trans import config as cg
    import json
    from datetime import datetime
    def _exists_or_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
            return False
        else:
            return True

    configs = {key: value for key, value in cg.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    json_name = '{}-{}-{}.json'.format(current_time, configs['net_setting']['SR_model'],configs['net_setting']['Recon_model'])
    jason_path = os.path.join(save_folder, json_name)
    with open(jason_path, 'w') as f:
        json.dump(configs, f)
    print('[*] config saved as {}'.format(json_name))
def fft(im):
    """
    Params:
        -im:  ndarray in shape [height, width, channels]

    """
    assert len(im.shape) == 3
    spec = np.fft.fft2(im, axes=(0, 1))
    return np.fft.fftshift(spec, axes=(0, 1))

def spectrum2im(fs):
    """
    Convert the Fourier spectrum into the original image
    Params:
        -fs: ndarray in shape [batch, height, width, channels]
    """
    fs = np.fft.fftshift(fs, axes=(1, 2))
    return np.fft.ifft2(fs, axes=(1, 2))



def get_2d_lf(filename, path, normalize_fn,**kwargs):
    def _LFP2ViewMap(img,angRes):
        img=np.squeeze(img)
        h,w=img.shape
        base_h = h // angRes
        base_w = w // angRes
        VP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                VP_[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w]= img[v::angRes, u::angRes]
        return VP_
    def _ViewMap2LFP(img,angRes):
        img=np.squeeze(img)
        h,w=img.shape
        base_h = h // angRes
        base_w = w // angRes
        LFP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                LFP_[v::angRes, u::angRes] = img[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w]
        return LFP_
    def _identity(img,angRes):
        return img
    image = imageio.imread(path + filename).astype(np.uint16)
    if 'read_type' in kwargs:
        read_type = kwargs['read_type']
    else:
        read_type = None

    if read_type is not None:
        assert 'ViewMap' in read_type or 'LFP' in read_type,'wrong img type'
        if '1' in read_type:
            trans_func=_identity if 'LFP' in read_type else _ViewMap2LFP
        elif '2' in read_type:
            trans_func = _identity if 'ViewMap' in read_type else _LFP2ViewMap
        else:
            raise Exception('wrong img type')
        image = trans_func(image,angRes=kwargs['angRes'])

    image = image[:, :, np.newaxis] if image.ndim == 2 else image
    return normalize_fn(image)

def get_img3d_fn(filename, path, normalize_fn):
    """
    Parames:
        mode - Depth : read 3-D image in format [depth=slices, height, width, channels=1]
               Channels : [height, width, channels=slices]
    """
    image = imageio.volread(path + filename) # [depth, height, width]
    # image = image[..., np.newaxis] # [depth, height, width, channels]
            
    return normalize_fn(image)
    
def rearrange3d_fn(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """
    
    image = np.squeeze(image) # remove channels dimension
    #print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth]) 
    for d in range(depth):
        image_re[:,:,d] = image[d,:,:]
    return image_re


def rearrange3d_fn_inverse(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """
    image = np.squeeze(image)  # remove channels dimension
    # print('reshape : ' + str(image.shape))
    height, width,depth = image.shape
    image_re = np.zeros([depth,height, width])
    for d in range(depth):
        image_re[d, :, :]=image[:, :, d]
    return image_re

def get_and_rearrange3d(filename, path, normalize_fn,**kwargs):
    image = get_img3d_fn(filename, path, normalize_fn=normalize_fn)
    return rearrange3d_fn(image)
    
def get_img2d_fn(filename, path, normalize_fn, **kwargs):
  
    image = imageio.imread(path + filename).astype(np.uint16)
    if image.ndim == 2:
        image = image[:,:, np.newaxis]
    #print(image.shape)
    return normalize_fn(image, **kwargs)

def get_lf_extra(filename, path, n_num, normalize_fn, padding=False, **kwargs):
    image = get_img2d_fn(filename, path, normalize_fn, **kwargs)
    extra = lf_extract_fn(image, n_num=n_num, padding=padding)
    return extra



def normalize(x):  
    max_ = np.max(x)*1.1
    #max_ = 255.
    #max_ = np.max(x)
    x = x / (max_ / 2.)
    x = x - 1
    return x


def normalize_constant(im):
    assert im.dtype in [np.uint8, np.uint16]
    x = im.astype(np.float)
    max_ = 255. if im.dtype == np.uint8 else 65536.
    # x = x / (max_ / 2.) - 1.
    x = x / (max_)
    return x


def z_score(x):
    std_=np.std(x)
    mean_=np.mean(x)
    return (x-mean_)/std_

def min_max(x,eps=1e-7):

    max_ = np.max(x)
    min_ =np.min(x)
    return (x-min_)/(max_-min_+eps)


def normalize_percentile(im, low=0.2, high=100,clip=True,is_random=False):
    if is_random:
        _p_low  = np.random.uniform(0.1,0.5)
        p_low  = np.percentile(im,_p_low)

        _p_high = np.random.uniform(99.5,99.9)
        p_high = np.percentile(im,_p_high)
    else:
        p_low = np.percentile(im, low)
        p_high = np.percentile(im, high)
    eps = 1e-7
    x = (im - p_low) / (p_high - p_low + eps)
    if clip:
        #x[x>1.0]=1.0
        x[x<.0]=.0
    #print('%.2f-%.2f' %  (np.min(x), np.max(x)))
    return x

def normalize_percentile_z_score(im, low=0.2, high=99.8):
    p_low = np.percentile(im, low)
    p_high = np.percentile(im, high)
    eps = 1e-7
    x = np.clip(im, p_low, p_high)
    mean_=np.mean(x)
    std=np.std(x)
    return (x-mean_)/std

def binary_normal(x):
    # max_ = np.max(x)
    max_ = 255.
    # max_ = np.max(x)
    x=x/max_

    return x
def resize_fn(x, size):
    '''
    Param:
        -size: [height, width]
    '''
    x = np.array(pilimg.fromarray(x).resize(size=(size[1], size[0]), resample=pilimg.BICUBIC))
    
    return x
    
def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection
    
    Params:
        -lf2d - 2-D light field projection
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                     False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel' 
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n*n, h, w, c]) # [depth, h, w, c]
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i : h : n, j : w : n, :] = lf2d[i : h : n, j : w : n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n*n])
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i : h : n, j : w : n, d] = lf2d[i : h : n, j : w : n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))

        if mode == 'toChannel':
            
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n*n])
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, : , d] = lf2d[i : h : n, j : w : n]
                    d += 1
                    
        elif mode == 'toDepth':
            lf_extra = np.zeros([n*n, new_h, new_w, c]) # [depth, h, w, c]
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i : h : n, j : w : n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
            
    return lf_extra
    
    
def do_nothing(x):
    return x

def normal_clip(x, low=1, high=100):
    min_ = np.percentile(x, low)
    max_ = np.max(x) if high == 100 else np.percentile(x, high)
    x = np.clip(x, min_, max_)

    # return binary_normal(x)
    return min_max(x)

def _write3d(x, path, bitdepth=8,clip=False):
    """
    x : [depth, height, width, channels=1]
    """
    assert (bitdepth in [8, 16, 32])
    max_ = 1.2*np.max(x)
    if clip:
        x=np.clip(x,0,max_)
    if bitdepth == 32:
         x = x.astype(np.float32)

    else:
        #x = _clip(x, 0.2)
        min_ = np.min(x)
        x = (x - min_) / (max_ - min_)
        if bitdepth == 8:
            x = x * 255
            x = x.astype(np.uint8)  
        else:
            x = x * 65535
            x = x.astype(np.uint16)
    imageio.volwrite(path, x[..., 0])
        
def write3d(x, path, bitdepth=32):
    """
    x : [batch, depth, height, width, channels] or [batch, height, width, channels>3]
    """

    dims = len(x.shape)
    


    x_re = np.transpose(np.transpose(x,[0,2,3,1]),[0,2,3,1])
    x_re =x_re[...,None]
    batch=x_re.shape[0]
    if batch == 1:
        _write3d(x_re[0], path, bitdepth)
    else:
        fragments = path.split('.')
        new_path = ''
        for i in range(len(fragments) - 1):
            new_path = new_path + fragments[i]
        for index, image in enumerate(x_re):
            #print(image.shape)
            _write3d(image, new_path + '_' + str(index) + '.' + fragments[-1], bitdepth)


def load_psf(path, n_num=11, psf_size=155, n_slices=16):
    '''
    Return : [n_num, n_num, n_slices, psf_size, psf_size, 1, 1]
    '''
    print('loading psf...')
    file_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
    if len(file_list) != n_num ** 2:
        raise Exception('psf files number must be euqal to Nnum^2');
        
    psf5d = np.zeros([n_num, n_num, n_slices, psf_size, psf_size])
    for i in range(n_num):
        for j in range(n_num):
            idx = i * n_num + j
            psf = imageio.volread(path + file_list[idx]) # [depth=n_slices, psf_size, psf_size]
            psf5d[i,j,...] = psf
            
    print('load psf5d in shape %s' % str(psf5d.shape))        
    return psf5d[..., np.newaxis, np.newaxis]  
    
def generate_mask(n_num, img_size):
    '''
    Generate a mask that help to extract different view from a 3-D scene. Used in forward projection.
    Return: mask[img_size, img_size, n_num, n_num]
    '''
    mask = np.zeros([img_size, img_size, n_num, n_num])
    for i in range(n_num):
        for j in range(n_num):
            for h in range(0, img_size):
                for w in range(0, img_size):
                    if h % n_num == i and w % n_num == j:
                        mask[h, w, i, j] = 1
                        
    return mask
    


# def get_laplace_pyr(img,layer_num=3):
#     batch,height,width,channel=img.shape
#     lap_batch=[]
#     def _gaussian(ori_image, down_times=3):
#         temp_gau = ori_image.copy()
#         gaussian_pyramid = [temp_gau]
#         for i in range(down_times):
#             temp_gau = cv2.pyrDown(temp_gau)
#             gaussian_pyramid.append(temp_gau)
#             temp = rearrange3d_fn_inverse(temp_gau).astype(np.float32, casting='unsafe')
#             imageio.volwrite('gau_%d.tif' % i, temp)
#
#         return gaussian_pyramid
#     def _laplacian(gaussian_pyramid, up_times=3):
#         laplacian_pyramid = [gaussian_pyramid[-1]]
#         for i in range(up_times, 0, -1):
#             temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
#             temp_lap = cv2.subtract(gaussian_pyramid[i - 1], temp_pyrUp)
#             laplacian_pyramid.append(temp_lap)
#         return laplacian_pyramid
#     for i in range(batch):
#         gp = _gaussian(img[i],layer_num-1)
#         lp = _laplacian(gp, layer_num - 1)
#         lap_batch.append(lp)
#     return lap_batch



