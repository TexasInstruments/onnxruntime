import os
import platform
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

if platform.machine() == 'aarch64':
    dataset_base = '/home/root'
    numImages = 100
else : 
    dataset_base = '/home/anand/workarea/deps/'
    numImages = 3

tidl_tensor_bits = 8
numFramesCalibration = 3
biasCalibrationIterations = 2
tidl_calibration_accuracy_level = 1
num_tidl_subgraphs = 16
debug_level = 0
power_of_2_quantization = 'no'
enable_high_resolution_optimization = 'no'
pre_batchnorm_fold = 1

tidl_tools_path = '../../../tidl/c7x-mma-tidl/tidl_tools/'

artifacts_folder = '../../../onnxrt-artifacts/'

required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder,
"import":'no'
}
optional_options = {
"tidl_platform":"J7",
"tidl_version":"7.2",
"tidl_tensor_bits":tidl_tensor_bits,
"debug_level":debug_level,
"num_tidl_subgraphs":num_tidl_subgraphs,
"tidl_denylist":"",
"tidl_calibration_accuracy_level":tidl_calibration_accuracy_level,
"tidl_calibration_options:num_frames_calibration": numFramesCalibration,
"tidl_calibration_options:bias_calibration_iterations": biasCalibrationIterations,
"power_of_2_quantization": power_of_2_quantization,
"enable_high_resolution_optimization": enable_high_resolution_optimization,
"pre_batchnorm_fold" : pre_batchnorm_fold,
"reserved" : 1601
}

#lables = '../testvecs/input/labels.txt'
lables = '../../../tidl/c7x-mma-tidl/ti_dl/test/testvecs/input/labels.txt'
models_base_path = '../../../..//onnx_models'

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def get_class_labels(output, org_image_rgb):
    output = np.squeeze(np.float32(output)) 
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)

    outputoffset = 0 if(output.shape[0] == 1001) else 1 
    top_k = output.argsort()[-5:][::-1]
    labels = load_labels(lables)
    for j, k in enumerate(top_k):
        curr_class = f'\n  {j}  {output[k]:08.6f}  {labels[k+outputoffset]} \n'
        classes = classes + curr_class if ('classes' in locals()) else curr_class 
    draw.text((0,0), classes, fill='red')
    source_img = source_img.convert("RGB")
    classes = classes.replace("\n", ",")
    return(classes, source_img)

colors_list = [
( 255, 	 0,	  0 ), ( 0	 , 255,    0 ), ( 0	,   0,	 255 ), ( 255, 255,	    0  ), ( 0	 , 255,  255  ), ( 255,   0,	 255  ),
( 255, 	 64,  0 ), ( 64	 , 255,    0 ), ( 64,   0,	 255 ), ( 255, 255,	   64  ), ( 64	 , 255,  255  ), ( 255,   64,	 255  ),
( 196, 	128,  0 ), ( 128 , 196,    0 ), ( 128,  0,	 196 ), ( 196, 196,	  128  ), ( 128	 , 196,  196  ), ( 196,   128,	 196  ),
( 64, 	128,  0 ), ( 128 , 64,     0 ), ( 128,  0,	 64  ), ( 196,   0,    0  ), ( 196	 ,  64,   64  ), ( 64,    196,	  64  ),
( 64,   255, 64 ), ( 64	 , 64,   255 ),( 255, 64,	 64  ), (128,  255,   128  ), ( 128	, 128,    255  ),( 255,   128,	 128  ),
( 196,  64, 196 ), ( 196, 196,    64 ),( 64,  196,	196  ), (196,  255,   196  ), ( 196	, 196,    255  ),( 196,   196,	 128  )]

def mask_transform(inp):
    colors = np.asarray(colors_list)
    inp = np.squeeze(inp)
    colorimg = np.zeros((inp.shape[0], inp.shape[1], 3), dtype=np.float32)
    height, width = inp.shape
    inp = np.rint(inp)
    inp = inp.astype(np.uint8)
    for y in range(height):
        for x in range(width):
            if(inp[y][x] < 22):
                colorimg[y][x] = colors[inp[y][x]]
    inp = colorimg.astype(np.uint8)
    return inp

def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(rgb,m)
    yuv[:,:, 1:] += 128.0
    rgb = np.clip(yuv, 0.0, 255.0)
    return yuv

def YUV2RGB( yuv ):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 2.0320025777816772],
                 [ 1.14019975662231445, -0.5811380310058594 , 0.00001542569043522235] ])
    yuv[:,:, 1:] -= 128.0
    rgb = np.dot(yuv,m)
    rgb = np.clip(rgb, 0.0, 255.0)
    return rgb

def seg_mask_overlay(output_data, org_image_rgb):
  classes = ''
  output_data = np.squeeze(output_data)
  if (output_data.ndim > 2) :
    output_data = output_data.argmax(axis=2)
  output_data = np.squeeze(output_data)
  mask_image_rgb  = mask_transform(output_data) 
  org_image  = RGB2YUV(org_image_rgb)
  mask_image = RGB2YUV(mask_image_rgb)
  
  org_image[:,:, 1] = mask_image[:,:, 1]
  org_image[:,:, 2] = mask_image[:,:, 2]
  blend_image = YUV2RGB(org_image)
  blend_image = blend_image.astype(np.uint8)
  blend_image = Image.fromarray(blend_image).convert('RGB')
  
  return(classes, blend_image)

def det_box_overlay(outputs, org_image_rgb):
    classes = ''
    source_img = org_image_rgb.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for i in range(int(outputs[3][0])):
        if(outputs[2][0][i] > 0.1) :
            ymin = outputs[0][0][i][0]
            xmin = outputs[0][0][i][1]
            ymax = outputs[0][0][i][2]
            xmax = outputs[0][0][i][3]
            draw.rectangle(((int(xmin*source_img.width), int(ymin*source_img.height)), (int(xmax*source_img.width), int(ymax*source_img.height))), outline = colors_list[int(outputs[1][0][i])%len(colors_list)], width=2)
    
    source_img = source_img.convert("RGB")
    return(classes, source_img)


mlperf_models_configs = {
    'squeezenet' : {
        'model_path' : '../../csharp/testdata/squeezenet.onnx',
        'dataset_list' : os.path.join(dataset_base,'tflite-test-data/tidl-dataset-lite/imagenet_1000/val_1000.txt'),
        'mean': [0, 0, 0],
        'std' : [1, 1, 1],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
    },
    'squeezenet1.1' : {
        'model_path' : os.path.join(models_base_path, 'squeezenet1.1.onnx'),
        'dataset_list' : os.path.join(dataset_base,'tflite-test-data/tidl-dataset-lite/imagenet_1000/val_1000.txt'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
    },
    'mobilenetv2-1.0' : {
        'model_path' : os.path.join(models_base_path, 'mobilenetv2-1.0.onnx'),
        'dataset_list' : os.path.join(dataset_base,'tflite-test-data/tidl-dataset-lite/imagenet_1000/val_1000.txt'),
        'mean': [123.675, 116.28, 103.53],
        'std' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'model_type': 'classification'
   },
    'deeplabv3_mnv2_ade20k_float' : {
        'model_path' : os.path.join(models_base_path,'deeplabv3_mnv2_ade20k_float.tflite'),
        'dataset_list' : os.path.join(dataset_base,'tflite-test-data/tidl-dataset-lite/ADEChallengeData2016Val/seg_val_list.txt'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 32,
        'model_type': 'seg'
    },
    'ssd_mobilenet_v1_coco_2018_01_28' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v1_coco_2018_01_28_th_0p3.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od'
    },
    'ssd_mobilenet_v2_coco_2018_03_29' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_coco_2018_03_29.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od'
    },
    'ssd_mobilenet_v2_300_float' : {
        'model_path' : os.path.join(models_base_path,'ssd_mobilenet_v2_300_float.tflite'),
        'mean': [127.5, 127.5, 127.5],
        'std' : [1/127.5, 1/127.5, 1/127.5],
        'num_images' : numImages,
        'num_classes': 91,
        'model_type': 'od'
    },
}
