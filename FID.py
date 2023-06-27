import tensorflow as tf
import torch 
from model import *
from tqdm import tqdm
import numpy as np
import math
from train import *
from parameter import get_args, defaults
import scipy
from keras.datasets import cifar10
from keras import backend as K
BATCH_SIZE = 256
num = 10000
latent_dim = 10
model_sigma = False
decoder_type = 'Gaussian'

inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                              weights="imagenet", 
                              pooling='avg', input_shape=(299, 299, 3))

dataset = 'CIFAR10'

def compute_embeddings(dataloader, count):
    image_embeddings = []

    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)

def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


count = math.ceil(num/BATCH_SIZE)

(x_train, y_train), (x_test, y_test) = cifar10.load_data() # fashion_mnist.load_data()


x_test = x_test.astype('float32')
y_test = tf.one_hot(y_test, depth=10)
y_test = tf.squeeze(y_test, axis=1)
print(x_test.shape)
print(y_test.shape)

# 取出num張真實圖片
real_data = x_test / 255.
# sampling 雜訊
gen_input = K.random_normal(shape=(num, latent_dim, ),mean=0., stddev=0.5)
# 加入label資訊
gen_input = tf.concat([gen_input, y_test], axis=-1)
print("real_data:", real_data.shape)
print("gen_input:", gen_input.shape)

gen_input = torch.tensor(gen_input.numpy())
print("real_data:", real_data.shape)
print("gen_input:", gen_input.shape)

cvae = CVAE(latent_dim, dataset, decoder_type)
#Decoder = GaussianDecoder(latent_dim, dataset, model_sigma)
cvae = torch.load(".\saved_model\CIFAR10-Gaussian-e100-z10-Jun-05-21-09-PM")
summary(cvae)

with torch.no_grad():
    output = [t.numpy() for t in cvae.decoder(gen_input)]

gen_data = []
for i in output:
    for j in i:
        gen_data.append(j)

gen_data = np.array(gen_data)
gen_data = torch.tensor(gen_data)
gen_data  = gen_data.reshape((gen_data.shape[0], 32, 32, 3))

trainloader = tf.data.Dataset.from_tensor_slices(real_data).batch(BATCH_SIZE)
genloader = tf.data.Dataset.from_tensor_slices(gen_data).batch(BATCH_SIZE)
trainloader = trainloader.map(lambda x: tf.image.resize(x , (299, 299)))
genloader = genloader.map(lambda x: tf.image.resize(x, (299, 299)))

real_image_embeddings = inception_model.predict(trainloader)
generated_image_embeddings = inception_model.predict(genloader)
print(real_image_embeddings.shape, generated_image_embeddings.shape)
# real_image_embeddings.shape, generated_image_embeddings.shape

fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
print(fid)