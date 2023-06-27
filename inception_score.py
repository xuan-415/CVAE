from keras import backend as K
import numpy as np, cv2
import tensorflow as tf
from model import *

def KL(P, Q):
    divergence = np.sum(P*np.log(P/Q))
    return divergence

def inception_score(imgs, splits=10):
    split_scores = []

    for k in range(splits):
        part = imgs[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(KL(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

N = 1000
classes = 10
BATCH_SIZE = 128
latent_dim = 10
dataset = 'CIFAR10'

# load model
cvae = CVAE(latent_dim, dataset, False)
# Decoder.summary()
cvae = torch.load(".\saved_model\CIFAR10-Gaussian-e100-z10-Jun-05-21-09-PM")


# generate images
gen_input = K.random_normal(shape=(N, latent_dim, ),mean=0., stddev=0.5)

# 加入label資訊
labels = np.zeros((N, 10))
for i in range(10):
    label = np.zeros((int(N / classes), 10))
    label[:, i] = 1
    labels[i * (N // classes): (i+1) * (N // classes), :] = label

print("labels:", labels.shape)
print("gen_input:", gen_input.shape)
gen_input = tf.concat([gen_input, labels], axis=-1)
gen_input = torch.tensor(gen_input.numpy())
print("gen_input:", gen_input.shape)

with torch.no_grad():
    output = [t.numpy() for t in cvae.decoder(gen_input)]

gen_data = []
for i in output:
    for j in i:
        gen_data.append(j)

gen_data = np.array(gen_data)
gen_data = torch.tensor(gen_data)
gen_data = gen_data.reshape((gen_data.shape[0], 32, 32, 3))
print(gen_data.shape)
print(gen_data[0].dtype)
genloader = tf.data.Dataset.from_tensor_slices(gen_data)
genloader = genloader.map(lambda x: tf.image.resize(x, (299, 299)))
genloader = genloader.map(lambda x: x / 255.0).shuffle(1000).batch(BATCH_SIZE)
# gen_data  = cv2.resize(gen_data , (299, 299))
# print(resized_data.shape)

inception_model = tf.keras.applications.InceptionV3(weights="imagenet") 
# inception_model.summary()
preds = inception_model.predict(genloader)
print(preds.shape)

# compute inception score
scores, std = inception_score(preds)
print(scores, std)
print("Inception Score : {score} ± {std}".format(score=scores, std=std))
