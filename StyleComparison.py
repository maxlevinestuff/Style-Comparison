import os
import tensorflow as tf
import PIL

import imageio
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import statistics
import random

import keras.preprocessing.image
from keras.applications.imagenet_utils import preprocess_input

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import seaborn as sns; sns.set_theme()
sns.set(font_scale = 0.75)

category_batch_size = 3

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    short_dim = min(shape)
    
    if int(long_dim.numpy()) > 2000:
        max_dim = 2000
    elif int(long_dim.numpy()) < 500:
        max_dim = 500
    else:
        max_dim = int(long_dim.numpy())
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img#int(min(new_shape).numpy())

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def image_similarity(img1, img2):
    style_extractor = vgg_layers(style_layers)
    img1_style_output = [gram_matrix(style_output) for style_output in style_extractor(img1*255)]
    img1_gram = img1_style_output[4].numpy().flatten()
    img2_style_output = [gram_matrix(style_output) for style_output in style_extractor(img2*255)]
    img2_gram = img2_style_output[4].numpy().flatten()

    dot_product = np.dot(img1_gram, img2_gram)
    norm_product = np.linalg.norm(img1_gram)*np.linalg.norm(img1_gram)
    norm_to_dot_product = dot_product / norm_product
    return norm_to_dot_product


# Load images in the images folder into array
cwd_path = os.getcwd()
data_path =cwd_path + '/images'
data_dir_list = os.listdir(data_path)
data_dir_list = sorted(data_dir_list) #sort alphabetically
print(data_dir_list)
img_data_list=[]
for dataset in data_dir_list:
        img_path = data_path + '/'+ dataset
        print(img_path)
        img_data_list.append(load_img(img_path))

print(image_similarity(img_data_list[0], img_data_list[1]))

heatmap_data = np.zeros((len(img_data_list), len(img_data_list)))
print(heatmap_data)

for x in range(0, len(img_data_list)):
    for y in range(0, len(img_data_list)):
        print(str(x) + ", " + str(y))
        heatmap_data[x][y] = image_similarity(img_data_list[x], img_data_list[y]) #uncomment this back
        #heatmap_data[x][y] = abs(heatmap_data[x][y] - 1) #make data show "distance from 1"

variance_per_batch = np.zeros((int(len(img_data_list) / category_batch_size), int(len(img_data_list) / category_batch_size)))
for x in range(0, len(variance_per_batch)):
    for y in range(0, len(variance_per_batch)):
        variance_per_batch[x][y] = statistics.variance(heatmap_data[(x*category_batch_size):(x*category_batch_size)+category_batch_size, (y*category_batch_size):(y*category_batch_size)+category_batch_size].flatten())

print(heatmap_data)

#pad with zeros to display on heatmap
m,n = variance_per_batch.shape

# new_array = np.array(["%.2f" % x for x in variance_per_batch.reshape(variance_per_batch.size)])
# new_array = new_array.reshape(variance_per_batch.shape)
# variance_per_batch = new_array

# remap1 = np.empty((m,3*n),dtype=variance_per_batch.dtype)
# remap1[:,::3] = variance_per_batch
# print(remap1)
# remap2=np.empty((len(remap1) * 3,len(remap1[0])), dtype=variance_per_batch.dtype)
# remap2[::3] = remap1
# print(remap2)
# print(remap2.shape)

# print("and")
# print(heatmap_data)
# print(heatmap_data.shape)

print(variance_per_batch)

plt.figure(figsize=(16,12))
heatmap = sns.heatmap(heatmap_data, xticklabels=data_dir_list, yticklabels=data_dir_list)
#heatmap = sns.heatmap(heatmap_data, annot=remap2, fmt='', xticklabels=data_dir_list, yticklabels=data_dir_list)
heatmap_figure = heatmap.get_figure()

#img = [plt.imread("images/adventure1.png"), plt.imread("images/adventure2.png")]
img = [plt.imread("images/" + image) for image in data_dir_list]
ax = plt.gca()

y_tick_labels = ax.yaxis.get_ticklabels()
x_tick_labels = ax.xaxis.get_ticklabels()

random.seed(10)
count = 0
for i,im in enumerate(img):
    if (count % category_batch_size == 0):
        color = [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]
    count += 1
    ib = OffsetImage(im, zoom=.2)
    ib.image.axes = ax
    ab = AnnotationBbox(ib,
                    y_tick_labels[i].get_position(),
                    frameon=True,
                    box_alignment=(1, .5),
                    bboxprops =dict(edgecolor=color  ,  lw = 3)
                    )
    ax.add_artist(ab)
    ab = AnnotationBbox(ib,
                    x_tick_labels[i].get_position(),
                    frameon=True,
                    box_alignment=(.5, 0),
                    bboxprops =dict(edgecolor=color  ,  lw = 3)
                    )
    ax.add_artist(ab)

ax.hlines([n for n in range(1, len(img_data_list)) if n % category_batch_size == 0], *ax.get_xlim(), color='white', lw=3)
ax.vlines([n for n in range(1, len(img_data_list)) if n % category_batch_size == 0], *ax.get_ylim(), color='white', lw=3)

for x in range(0, len(variance_per_batch)):
    for y in range(0, len(variance_per_batch)):
        #plt.annotate("{:.2f}".format(variance_per_batch[y][x]), ((x)*3 + 1.5,(y+1)*3 - 1.5), ha='center', va='center')
        ax.text((x)*3 + 1.5,(y+1)*3 - 1.5, "{:.2f}".format(variance_per_batch[y][x]), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'), ha='center', va='center')

sns.set(font_scale = 1.1)
plt.title("Heatmap Comparison of Style Similarity of Grouped Animation Styles with Variance Shown Between Styles", y=1.1)
sns.set(font_scale = 1)
ax.text((len(variance_per_batch) / 2)*3, (len(variance_per_batch) / 2)*3 - 1, "{:.2f}".format(statistics.variance(heatmap_data.flatten())), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'), ha='center', va='center')
#ax.text(((len(variance_per_batch) / 2)*3, (len(variance_per_batch) / 2)*3), "{:.2f}".format(statistics.variance(heatmap_data.flatten())), color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'), ha='center', va='center')

#plt.text(2, 2, "An annotation", horizontalalignment='center', verticalalignment='center', size='medium', color='black', weight='semibold')
plt.xticks([])
plt.yticks([])

heatmap_figure.savefig("heatmap.png",bbox_inches='tight')

plt.clf()
# create a dataset
height = []
bars_names = []
colors = []

for x in range(0, len(variance_per_batch)):
    for y in range(0, len(variance_per_batch)):
        if not (str(x) + " vs. " + str(y)) in bars_names and not (str(y) + " vs. " + str(x)) in bars_names:
            height.append(variance_per_batch[y][x])
            bars_names.append(str(x) + " vs. " + str(y))
            if x == y:
                colors.append('red')
            else:
                colors.append('blue')

height.append(statistics.variance(heatmap_data.flatten()))
bars_names.append("All")
colors.append('green')

# create a dataset
x_pos = np.arange(len(bars_names))

# Create bars
plt.bar(x_pos, height, color=colors)

# Create names on the x-axis
plt.xticks(x_pos, bars_names)

ax = plt.gca()
sns.set(font_scale = 2.5)
plt.title("Variance in Style Similarity Between Various Animation Stills")
sns.set(font_scale = 1)
ax.set_ylabel('Variance')
x_tick_labels = ax.xaxis.get_ticklabels()
for i in range(0, len(img)):
    for j in range(0, category_batch_size):
        ib = OffsetImage(img[int(bars_names[i][0])*3 + j], zoom=.2)
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        x_tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment=(.5 + (j*0.1), 1 + (j*0.1)),
                        bboxprops =dict(edgecolor='white'  )
                        )
        ax.add_artist(ab)
    for j in range(0, category_batch_size):
        ib = OffsetImage(img[int(bars_names[i][6])*3 + j], zoom=.2)
        ib.image.axes = ax
        ab = AnnotationBbox(ib,
                        x_tick_labels[i].get_position(),
                        frameon=False,
                        box_alignment=(.5 + (j*0.1), 2 + (j*0.1)),
                        bboxprops =dict(edgecolor='white' )
                        )
        ax.add_artist(ab)


# Show graph
plt.savefig("bar_chart.png",bbox_inches='tight')