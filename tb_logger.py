from __future__ import print_function
from os.path import join

try:
    import tensorflow as tf
except:
    print('Tensorflow could not be imported, therefore tensorboard cannot be used.')

from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime

class TBLogger(object):

    def __init__(self, log_dir, folder_name = '' ):

        self.log_dir = join(log_dir, folder_name + ' ' + datetime.datetime.now().strftime("%I%M%p, %B %d, %Y"))
        self.log_dir  = self.log_dir.replace('//','/')
        self.writer = tf.summary.create_file_writer(self.log_dir)

    #Add scalar
    def log_scalar(self, tag, value, step=0):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def make_list_of_2D_array(self, im):
        if type(im) == type([]):
            return im
        ims = []
        if len(im.shape) == 2:
            ims.append(im)
        elif len(im.shape) == 3:
            for i in range(im.shape[0]):
                ims.append(np.squeeze(im[i,:,:]))

        elif len(im.shape) == 4:
            for i in range(im.shape[0]):
                ims.append(np.squeeze(im[i, 0, :, :]))
        return ims

    def log_images(self, tag, images, step=0, dim = 2, max_imgs = 50,cm='jet'):

        #Make sure images are on numpy format in case the input is a Torch-variable
        images = self.convert_to_numpy(images)

        try:
            if len(images.shape)>2:
                dim = 3
        except:
            None

        #Make list of images
        if dim == 2:
            images = self.make_list_of_2D_array(images)

        #If 3D we make one list for each slice-type
        if dim == 3:
            new_images_ts, new_images_il, new_images_cl = self.get_slices_from_3D(images)
            self.log_images(tag + '_timeslice', new_images_ts, step, 2, max_imgs)
            self.log_images(tag + '_inline', new_images_il, step, 2, max_imgs)
            self.log_images(tag + '_crossline', new_images_cl, step, 2, max_imgs)
            return


        im_summaries = []

        for nr, img in enumerate(images):

            #Grayscale
            if cm == 'gray' or cm == 'grey':
                img = img.astype('float')
                img = np.repeat(np.expand_dims(img,2),3,2)
                img -= img.min()
                img /= img.max()
                img *= 255
                img = img.astype('uint8')

            # Write the image to a string
            s = BytesIO()
            plt.imsave(s, img, format='png')

            # Create a 4D tensor for the image (batch_size, height, width, channels)
            img_tensor = np.expand_dims(img, axis=0)  # Adding batch dimension


            # Log the image summary
            with self.writer.as_default():
                print(f"Image dimensions: {img_tensor.shape}")
                tf.summary.image('%s/%d' % (tag, nr), img_tensor, step=step)
                print('the function tf.summary.image has been used seccessfully')

            # If grayscale, add channels dimension
            if cm == 'gray' or cm == 'grey':
                img_tensor = np.expand_dims(img_tensor, axis=-1)  # Add channels dimension for grayscale images
                

            with self.writer.as_default():
                for img_sum in im_summaries:
                    print(f"Image dimensions: {img_tensor.shape}")
                    tf.summary.write(img_sum)
                    print('the function tf.summary.write has been used seccessfully')

    # Cuts out middle slices from image
    def get_slices_from_3D(self, img):

        new_images_ts = []
        new_images_il = []
        new_images_cl = []

        if len(img.shape) == 3:
            new_images_ts.append(np.squeeze(img[img.shape[0] / 2, :, :]))
            new_images_il.append(np.squeeze(img[:, img.shape[1] / 2, :]))
            new_images_cl.append(np.squeeze(img[:, :, img.shape[2] / 2]))

        elif len(img.shape) == 4:
            for i in range(img.shape[0]):
                new_images_ts.append(np.squeeze(img[i, img.shape[1] / 2, :, :]))
                new_images_il.append(np.squeeze(img[i, :, img.shape[2] / 2, :]))
                new_images_cl.append(np.squeeze(img[i, :, :, img.shape[3] / 2]))

        elif len(img.shape) == 5:
            for i in range(img.shape[0]):
                new_images_ts.append(np.squeeze(img[i, 0, img.shape[2] / 2, :, :]))
                new_images_il.append(np.squeeze(img[i, 0, :, img.shape[3] / 2, :]))
                new_images_cl.append(np.squeeze(img[i, 0, :, :, img.shape[4] / 2]))

        return new_images_ts, new_images_il, new_images_cl
    #Convert torch to numpy
    def convert_to_numpy(self,im):
        if type(im) == torch.autograd.Variable:
            #Put on CPU
            im = im.cpu()
            #Get np-data
            im = im.data.numpy()
        return im