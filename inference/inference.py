import tensorflow as tf
import numpy as np
import argparse
import os,sys
import math
import random
import cv2
import matplotlib.image as mpimg
from type import category

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def inference(input=0,inputType=1):
    slim = tf.contrib.slim
    sys.path.append('../')
    from nets import ssd_vgg_300, ssd_common, np_methods
    from preprocessing import ssd_vgg_preprocessing
    from notebooks import visualization
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # Restore SSD model.
    ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    # Main image processing routine.
    def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                  feed_dict={img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes

    # input is a image
    inputType = int(inputType)
    if inputType is 1:
        if input == 0:
            print("At least indicate 1 input video")
            exit(-1)
        # Test on some demo image and visualize output.
        img = mpimg.imread(input)
        rclasses, rscores, rbboxes = process_image(img)

        # Find the name of the category num
        print(list(map(lambda i:"{}:{}".format(i,category[i]),list(rclasses))))
        rclasses = np.array(list(map(lambda i:"{}:{}".format(i,category[i]),list(rclasses))))

        # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        # plot the image directly
        visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    elif inputType == 2:
        # input is the video
        # plot the boxes into the image
        cap = cv2.VideoCapture(input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        #fourcc = cv2.CAP_PROP_FOURCC(*'CVID')
        print('fps=%d,size=%r,fourcc=%r'%(fps,size,fourcc))
        delay=10/int(fps)
        print(delay)
        if delay <= 1:
            delay = 1
        while (cap.isOpened()):
            ret, frame = cap.read()
            print(ret)
            if ret == True:
                image = frame
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = image
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                rclasses, rscores, rbboxes = process_image(image_np)

                #print(list(map(lambda i: "{}:{}".format(i, category[i]), list(rclasses))))
                rclasses = np.array(list(map(lambda i: "{}:{}".format(i, category[i]), list(rclasses))))

                # Visualization of the results of a detection.
                visualization.bboxes_draw_on_img(image_np, rclasses, rscores, rbboxes)
                cv2.imshow('frame', image_np)
                #cv2.waitKey(np.uint(delay))
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
                print('Ongoing...')
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    elif inputType ==3:
        print("save video")
        if input == 0:
            print("At least indicate 1 input video")
            exit(-1)
        def save_image(image_np):
            rclasses, rscores, rbboxes = process_image(image_np)
            # print(list(map(lambda i: "{}:{}".format(i, category[i]), list(rclasses))))
            rclasses = np.array(list(map(lambda i: "{}:{}".format(i, category[i]), list(rclasses))))
            visualization.bboxes_draw_on_img(image_np, rclasses, rscores, rbboxes)
            return image_np

        from moviepy.editor import VideoFileClip
        cap = cv2.VideoCapture(input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        cv2.destroyAllWindows()

        video = VideoFileClip(input)
        result = video.fl_image(save_image)
        output = os.path.join("./videos/output_{}".format(input.split("/")[-1]))
        result.write_videofile(output, fps=fps)
    else:
        cap = cv2.VideoCapture(0)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            #cv2.imshow('frame', frame)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(frame, axis=0)
            # Actual detection.
            rclasses, rscores, rbboxes = process_image(frame)

            # print(list(map(lambda i: "{}:{}".format(i, category[i]), list(rclasses))))
            rclasses = np.array(list(map(lambda i: "{}:{}".format(i, category[i]), list(rclasses))))
            # Visualization of the results of a detection.
            visualization.bboxes_draw_on_img(frame, rclasses, rscores, rbboxes)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_image", help="the path of the input image", dest='input',
                        default=os.path.join(sys.path[0], "images/mouse.jpeg"))
    parser.add_argument('-t', "--input_type", help="the type of the input: image or video", dest='type',
                        default=1)
    args = parser.parse_args()
    inference(args.input,args.type)
