# VRDL_HW2

## File description
 - `main.py`: Main file contains training loop and log infomation.
 - `dataloader.py`: Load data from TFRecord and form tf.data.Dataset.
 - `dcgan.py`: Modified DCGAN model.
 - `solver.py`: Define loss functions and optimizers.
 - `inference.py`: Inference the model and make 500 samples.
 - `helper.py`: Provided by TA.
 - `pick_front.py`: Select front-face from dataset.
 - `haarcascade_frontalface_default.xml`: Haarcascade frontalface model
 - `front_face.txt`: Results select by `pick_front.py`
 - `process_data.py`: Read original dataset and biuld TFRecord.