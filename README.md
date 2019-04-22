# tf-detection-example
A simple example to how to use Tensorflow detection API on a custom dataset


## Dataset
Download and install [labelImg tool](https://github.com/tzutalin/labelImg) 

- Put your images in `dataset/train/images` folder
- Create a `dataset/train/annotations/` folder
- `Open Dir` in labelImg tool to `dataset/train/images` folder
- `Change Save Dir` in labelImg tool to `dataset/train/annotations/` folder
- Start boring annotation task :woman_shrugging:

For the laziest, you can also use an existing dataset. 
Use the script `convert_dataset_format.py` to convert from a YOLO dataset to a PASCAL VOC style dataset.

## Install Tensorflow Object Detection
Download and install the [tensorflow object detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) preferably in a conda env
```bash
conda create --name tf_detection python=3.7 -y
conda activate tf_detection
```
Do not forget to add it to the `PYTHONPATH`

Set the `TF_MODELS_PATH` env var to the Tensorflow detection folder.
If you cloned in your home directory `export TF_MODELS_PATH=~/models` 
or cd into the directory and do `export TF_MODELS_PATH=$(pwd)`

## Generate tfr

Split test train
```bash
python split_train_test.py
```

Create the TFRecords
```bash
python create_pascal_tf_record.py --output_path=./train.record --data_dir=dataset/train
python create_pascal_tf_record.py --output_path=./test.record --data_dir=dataset/test
```

## Download a pre-trained model
The model used in this example is the `ssdlite_mobilenet_v2_coco_2018_05_09`, you can change it with whatever you like
```
python download_model.py
```
If you changed the model, download also the adequate pipeline config [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)


## Training
Run this command to begin the training
 ```bash
python ${TF_MODELS_PATH}/research/object_detection/model_main.py \
    --pipeline_config_path=ssdlite_mobilenet_v2_coco_modified.config \
    --model_dir=./finetuned_model \
    --alsologtostderr \
    --num_train_steps=30000 \
    --num_eval_steps=100
```

You can follow the training using tensorboard
```
tensorboard --logdir=finetuned_model
```

## Export graph
After the training completion, export the graph for inference

```bash
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=ssdlite_mobilenet_v2_coco_modified.config
TRAINED_CKPT_PREFIX=finetuned_model/model.ckpt-30000
EXPORT_DIR=finetuned_model/export_frozen
python ${TF_MODELS_PATH}/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```

## Inference
TODO
