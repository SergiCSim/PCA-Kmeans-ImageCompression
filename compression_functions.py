# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:19:54 2023

@author: sergi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
#from skimage.metrics import peak_signal_noise_ratio
from skimage import io


def plot_images(original_image, compressed_image):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(compressed_image)
    axes[1].set_title('Compressed Image')
    axes[1].axis('off')

    plt.show()
    
def plot_image(image):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')
    

def entropy(array):
    a = array.flatten()
    _, counts = np.unique(a, return_counts=True)
    probabilities = counts / len(a)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def rmse(image1, image2):
    return np.sqrt(((image1 - image2) ** 2).sum() / image1.size)

def psnr(image1, image2, max_pixel=None):
    if max_pixel is None:
        max_pixel = max(image1.max(), image2.max())
    return 20 * np.log10(max_pixel / rmse(image1, image2))


def compress_pca(image, n_components):
    
    max_pixel = image.max()
    
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    pca1 = PCA(n_components=n_components)
    pca2 = PCA(n_components=n_components)
    pca3 = PCA(n_components=n_components)
    
    compressed_R = pca1.fit_transform(R)
    compressed_G = pca2.fit_transform(G)
    compressed_B = pca3.fit_transform(B)
    
    compressed_R_inv = pca1.inverse_transform(compressed_R)
    compressed_G_inv = pca2.inverse_transform(compressed_G)
    compressed_B_inv = pca3.inverse_transform(compressed_B)
    
    compressed_image = np.stack([compressed_R_inv, compressed_G_inv, compressed_B_inv], axis=-1)
    compressed_image = np.array(compressed_image, dtype=image.dtype)
    compressed_image = np.clip(compressed_image, 0, max_pixel)
    
    output = {'compressed image': compressed_image,
              'entropy': entropy(compressed_image),
              'psnr': psnr(image, compressed_image, max_pixel=max_pixel)}
    
    return output


def find_best_n_components(image, threshold=0.95):

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    pca1 = PCA()
    pca2 = PCA()
    pca3 = PCA()
    
    pca1.fit_transform(R)
    pca2.fit_transform(G)
    pca3.fit_transform(B)
    
    cumulative_variance1 = np.cumsum(pca1.explained_variance_ratio_)
    cumulative_variance2 = np.cumsum(pca2.explained_variance_ratio_)
    cumulative_variance3 = np.cumsum(pca3.explained_variance_ratio_)

    return np.argmax((cumulative_variance1 > threshold) & (cumulative_variance2 > threshold) & (cumulative_variance3 > threshold))


def compress_kmeans(image, n_colors):
    
    max_pixel = image.max()
    
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    points = np.stack((R.flatten(), G.flatten(), B.flatten()), axis=1)
    
    image_array_sample = shuffle(points, random_state=0, n_samples=1_000)
    
    kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=0)
    kmeans.fit(image_array_sample)
    
    labels = kmeans.predict(points)
    
    compressed_image = np.array(kmeans.cluster_centers_[labels].reshape(image.shape), dtype=image.dtype)
    compressed_image = np.clip(compressed_image, 0, max_pixel)
    
    output = {'compressed image': compressed_image,
              'entropy': entropy(compressed_image),
              'psnr': psnr(image, compressed_image, max_pixel=max_pixel)}
    
    return output


def compress(image, n_components, n_colors, plot=True, info=True, name='image',
             original_path='images/original_images/', compression_path='images/compressed_images/'):
    
    output = {'original entropy': entropy(image),
              'original size': os.path.getsize(original_path + name + '.jpg')}
    if info:
        print('Size of the original file calculated')
    
    
    io.imsave(compression_path + name + '_original.jpg', image)
    output['original stored size'] = os.path.getsize(compression_path + name + '_original.jpg')
    if info:
        print('Size of the original stored image calculated')
    
    out = compress_pca(image, n_components)
    compressed_pca = out['compressed image']
    output['entropy pca'] = out['entropy']
    output['psnr pca'] = out['psnr']
    io.imsave(compression_path + name + '_pca.jpg', compressed_pca)
    output['pca stored size'] = os.path.getsize(compression_path + name + '_pca.jpg')
    if info:
        print('PCA compression done and compressed image stored as ' + name + '_pca.jpg')
    
    out = compress_kmeans(image, n_colors)
    compressed_kmeans = out['compressed image']
    output['entropy kmeans'] = out['entropy']
    output['psnr kmeans'] = out['psnr']
    io.imsave(compression_path + name + '_kmeans.jpg', compressed_kmeans)
    output['kmeans stored size'] = os.path.getsize(compression_path + name + '_kmeans.jpg')
    if info:
        print('K-means compression done and compressed image stored as ' + name + '_kmeans.jpg')
    
    out = compress_kmeans(compressed_pca, n_colors)
    compressed_image = out['compressed image']
    output['entropy'] = out['entropy']
    output['psnr'] = psnr(image, compressed_image, max_pixel=image.max()) #out['psnr']
    io.imsave(compression_path + name + '_compressed.jpg', compressed_image)
    output['compressed stored size'] = os.path.getsize(compression_path + name + '_compressed.jpg')
    if info:
        print('PCA + K-means compression done and compressed image stored as ' + name + '_compressed.jpg')
    
    if plot:
        plot_images(image, compressed_image)
        
    return compressed_image, output