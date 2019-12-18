from __future__ import print_function

import argparse
import os
import glob
import sys
import timeit
from tqdm import trange
import tensorflow as tf
import numpy as np
from scipy import misc
from model_mobilev2_shelf_2 import Segment2
from tools import decode_labels

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

sun_class = 14
SAVE_DIR = './evalModel/output/'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced Segment2")
    parser.add_argument("--img-path", type=str, default='./evalModel/test1/img-000001.jpg',
                        help="Path to the RGB image file or input directory.")
                        
    parser.add_argument("--model", type=str, default='others',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'])
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    parser.add_argument("--num-class",type=int,default=14,help="color 6 or 14",
                        choices=[6,14])
    parser.add_argument("--pb-path",type=str,default='./pb/SegmentModel.pb',help="Path of pb model",)                        
    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    img = misc.imread(img_path, mode='RGB')
    img = misc.imresize(img, [480, 640])

    img = img.astype(float)

    img = img[...,::-1]
    img -= IMG_MEAN

    print('input image shape: ', img.shape)

    return img, filename

def preprocess(img):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_r, img_g, img_b]), dtype=tf.float32)
    img -= IMG_MEAN
    
    img = tf.expand_dims(img, dim=0)

    return img

def check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[0:2]
    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h/32) + 1) * 32
        new_w = (int(ori_w/32) + 1) * 32
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
        
        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape

def main():
    args = get_arguments()    
    num_classes = sun_class
    show_classes = args.num_class
    output_graph_path=args.pb_path
    imgs = []
    filenames = []
    if os.path.isdir(args.img_path):
        file_paths = glob.glob(os.path.join(args.img_path, '*'))
        for file_path in file_paths:
            ext = file_path.split('.')[-1].lower()

            if ext == 'png' or ext == 'jpg':
                img, filename = load_img(file_path)
                imgs.append(img)
                filenames.append(filename)
    else:
        img, filename = load_img(args.img_path)
        img=preprocess(img)
        imgs.append(img)
        filenames.append(filename)

    shape = imgs[0].shape[0:2]    
    x = tf.placeholder(dtype=tf.float32, shape=img.shape)
    x, n_shape = check_input(x)


    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            img = sess.graph.get_tensor_by_name("Placeholder:0")            
            raw_output = sess.graph.get_tensor_by_name("final_out:0")           
            raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)
            raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
            raw_output_up = tf.argmax(raw_output_up, axis=3)
            pred = decode_labels(raw_output_up, shape, show_classes,num_classes)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for i in trange(len(imgs), desc='Inference', leave=True):
                start_time = timeit.default_timer()                           
                raw_output_up_temp = sess.run(raw_output, feed_dict={img: [imgs[i]]})
                preds = sess.run(pred, feed_dict={img: [imgs[i]]})
                misc.imsave(args.save_dir + filenames[i], preds[0])

if __name__ == '__main__':
    main()


