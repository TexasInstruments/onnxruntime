import numpy as np
import onnxruntime as rt
from PIL import Image
import time
import argparse
import os
import platform
from models import mlperf_models_configs, get_class_labels, seg_mask_overlay, det_box_overlay, required_options, optional_options, models_base_path

parser = argparse.ArgumentParser()
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
args = parser.parse_args()
#os.environ["TIDL_RT_PERFSTATS"] = "1"

TIDL_BASE_PATH = '/home/a0393754/work/onnxrt_7.3_rel/c7x-mma-tidl'
image_path = '../img/mushroom.png'
#image_path = '../../../tidl/c7x-mma-tidl/ti_dl/test/testvecs/input/airshow.jpg'
model_path = '../../csharp/testdata/squeezenet.onnx'
lables = '/home/a0393754/work/onnxrt_7.3_rel/c7x-mma-tidl/ti_dl/test/testvecs/input/labels.txt'


so = rt.SessionOptions()

print("Available execution providers : ", rt.get_available_providers())
print("Platform - ", platform.python_implementation())
#so.log_severity_level = 0
#so.log_verbosity_level = 4
#so.enable_profiling = True

delegate_options = {}
delegate_options.update(required_options)
delegate_options.update(optional_options)


calib_images = [os.path.join(TIDL_BASE_PATH,'ti_dl/test/testvecs/input/airshow.jpg'),
                os.path.join(TIDL_BASE_PATH,'ti_dl/test/testvecs/input/ADE_val_00001801.jpg')]
class_test_images = [os.path.join(TIDL_BASE_PATH,'ti_dl/test/testvecs/input/airshow.jpg')]
od_test_images    = [os.path.join(TIDL_BASE_PATH,'ti_dl/test/testvecs/input/ADE_val_00001801.jpg')]
seg_test_images   = [os.path.join(TIDL_BASE_PATH,'ti_dl/test/testvecs/input/ADE_val_00001801.jpg')]

def infer_image(sess, image_file, config):
  input_details = sess.get_inputs()
  input_name = input_details[0].name
  floating_model = (input_details[0].type == 'tensor(float)')
  height = input_details[0].shape[2]
  width  = input_details[0].shape[3]
  img    = Image.open(image_file).convert('RGB').resize((width, height))
  input_data = np.expand_dims(img, axis=0)
  input_data = np.transpose(input_data, (0, 3, 1, 2))

  if floating_model:
    input_data = np.float32(input_data)
    for mean, scale, ch in zip(config['mean'], config['std'], range(input_data.shape[1])):
        input_data[:,ch,:,:] = ((input_data[:,ch,:,:]- mean) * scale)
  
  start_time = time.time()
  #interpreter invoke call
  output = sess.run(None, {input_name: input_data})[0]
  #prof_file = sess.end_profiling()
  #print(prof_file)
  stop_time = time.time()
  infer_time = stop_time - start_time

  benchmark_dict = sess.get_TI_benchmark_data()
  print(benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start'])
  print(benchmark_dict)

  #outputs = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]
  return img, output, infer_time, 0

def run_model(model, mIdx, log_file):
    print("\n Running model  ", model)
    config = mlperf_models_configs[model]
    #set input images for demo
    config = mlperf_models_configs[model]
    if config['model_type'] == 'classification':
        test_images = class_test_images
    elif config['model_type'] == 'od':
        test_images = od_test_images
    elif config['model_type'] == 'seg':
        test_images = seg_test_images
    
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/'
    
    # delete the contents of this folder
    if args.compile:
        os.makedirs(delegate_options['artifacts_folder'], exist_ok=True)
        for root, dirs, files in os.walk(delegate_options['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    if(args.compile == True):
        delegate_options['import'] = 'yes'
        input_image = calib_images
    else:
        input_image = test_images
    
    numFrames = config['num_images']
    if(delegate_options['import'] == 'yes'):
        if numFrames > delegate_options['tidl_calibration_options:num_frames_calibration']:
            numFrames = delegate_options['tidl_calibration_options:num_frames_calibration']
    
    ############   set interpreter  ################################
    if args.disable_offload : 
        EP_list = ['CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] , providers=EP_list,sess_options=so)
    else:
        EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    ################################################################
    
    # run session
    for i in range(numFrames):
        #img, output, proc_time, sub_graph_time = infer_image(sess, input_image[i%len(input_image)], config)
        img, output, proc_time, sub_graph_time = infer_image(sess, input_image[i%len(input_image)], config)
        total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
        sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time
    
    #total_proc_time = total_proc_time/1000000
    total_proc_time = total_proc_time * 1000
    sub_graphs_time = sub_graphs_time/1000000

    # output post processing
    if(args.compile == False):  # post processing enabled only for inference
        if config['model_type'] == 'classification':
            classes, image = get_class_labels(output[0],img)
            print(classes)
        elif config['model_type'] == 'od':
            classes, image = det_box_overlay(output, img)
        elif config['model_type'] == 'seg':
            classes, image = seg_mask_overlay(output[0], img)
        else:
            print("Not a valid model type")

        #print("Saving image to ", delegate_options['artifacts_folder'])
        #image.save(delegate_options['artifacts_folder'] + "post_proc_out_"+os.path.basename(config['model_path'])+'_'+os.path.basename(input_image[i%len(input_image)]), "JPEG") 
    
    log = f'\n \n   #{mIdx+1:5d}, {model:50s}, Total time : {total_proc_time/(i+1):10.1f}, Offload Time : {sub_graphs_time/(i+1):10.1f} \n \n ' #{classes} \n \n'
    print(log) 
    log_file.write(log)


log_file = open("log.txt", "w+", buffering=1)

#models = mlperf_models_configs.keys()
#models=['mobilenetv2-1.0']
models = ['squeezenet']

log = f'Running {len(models)} Models - {models}\n'
print(log)
log_file.write(log)

for mIdx, model in enumerate(models):
    run_model(model, mIdx, log_file)

log_file.close()
