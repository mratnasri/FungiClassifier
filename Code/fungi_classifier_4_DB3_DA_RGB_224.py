# human_2D_classifier_15_dataAugmentation_ver3 with histogram equalization
import os
import os.path
import random
from sklearn.datasets import load_files
import numpy as np
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from numpy import mean, std
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, top_k_accuracy_score, ConfusionMatrixDisplay,  roc_auc_score, auc, roc_curve, RocCurveDisplay
import sys
import cv2
from glob import glob
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt

# 150x150 images
# 3 classes

#train_dir = '../../trial data/Training'
gram_img_dir = '../../Gram Stain Images_ver3_Crops'
gram_img_DA_dir = '../../Gram Stain Images_ver3_Crops_DA'
categories_n = 3
#classes = [x for x in range(200)]
labels = []
images = []
targets = []
noise_images = []
noise_targets = []
#IMG_SHAPE = (150, 150, 3)
IMG_SHAPE = (224, 224, 3)


def load_dataset(path, images, targets, label):
    img_names = glob(path)
    for fn in img_names:
        img = cv2.imread(fn)
        #img = load_img(fn)
        # print(img.shape)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        target = labels.index(label)  # \\ for windows
        targets.append(target)
    return


for label in os.listdir(gram_img_dir):
    labels.append(label)
    # train_categories.append(label)
    path = gram_img_dir + "/" + label + "/*.jpg"
    #train_n.append(len(os.listdir(gram_img_dir + "/" + label)))
    load_dataset(path, images, targets, label)

for label in os.listdir(gram_img_DA_dir):
    # labels.append(label)
    # train_categories.append(label)
    path = gram_img_DA_dir + "/" + label + "/*_cropped.jpg"
    #train_n.append(len(os.listdir(gram_img_dir + "/" + label)))
    load_dataset(path, images, targets, label)
    path = gram_img_DA_dir + "/" + label + "/*_rotated.jpg"
    load_dataset(path, images, targets, label)
    path = gram_img_DA_dir + "/" + label + "/*_noise.jpg"
    load_dataset(path, noise_images, noise_targets, label)

images_num = len(images)
noise_images_num = len(noise_images)
categories_n = len(labels)

print("Number of total samples = ", images_num)
print("Number of noise samples = ", noise_images_num)

# print(targets)

# convert to grayscale


"""def convert_gray(images):
    gray_images = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_img = cv2.equalizeHist(gray_img)
        gray_images.append(gray_img)
    return gray_images"""


#gray_images = convert_gray(images)

#cv2.imwrite("../../stretched.png", gray_images[2])
# split into training, testing and validation
mystate = 41
x_train, x_test, y_train, y_test = train_test_split(
    images, targets, test_size=0.3, shuffle=True, stratify=targets, random_state=mystate)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.5, shuffle=True, stratify=y_test, random_state=mystate)

x_train.extend(noise_images)
y_train.extend(noise_targets)
temp = list(zip(x_train, y_train))
random.shuffle(temp)
x_train, y_train = zip(*temp)

y_train_ohe = to_categorical(y_train, categories_n)
y_val_ohe = to_categorical(y_val, categories_n)


# preprocessing

#x_train = np.array(convert_img_to_array(x_train))
x_train = np.array(x_train)
print('Training set shape : ', x_train.shape)
#x_val = np.array(convert_img_to_array(x_val))
x_val = np.array(x_val)
print('Validation set shape : ', x_val.shape)
#x_test = np.array(convert_img_to_array(x_test))
x_test = np.array(x_test)
print('Test set shape : ', x_test.shape)

"""x_train = x_train.reshape((x_train.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1))
x_val = x_val.reshape((x_val.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1))
x_test = x_test.reshape((x_test.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1))"""

for img in x_train:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_val:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_test:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

#cv2.imwrite("../../stretched2.jpg", x_train[2])
# normalization
x_train = x_train.astype('float32')
x_train = x_train/255
x_val = x_val.astype('float32')
x_val = x_val/255
x_test = x_test.astype('float32')
x_test = x_test/255


def model_config(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    #model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    #model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(300, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(categories_n, activation='softmax'))

    model.summary()
    # compile the model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2)])
    return model

# evaluate model using 5-fold cross validation


def train_model(datax, datay, valx, valy):
    model = model_config(IMG_SHAPE)
    # fit model
    history = model.fit(datax, datay, epochs=30, batch_size=32,
                        validation_data=(valx, valy), verbose=2)
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['accuracy'])*100, std(history.history['accuracy'])*100, len(history.history['accuracy'])))
    print('Top-2 Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['top_k_categorical_accuracy'])*100, std(history.history['top_k_categorical_accuracy'])*100, len(history.history['top_k_categorical_accuracy'])))
    print('Validation Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['val_accuracy'])*100, std(history.history['val_accuracy'])*100, len(history.history['val_accuracy'])))
    print('Validation Top-2 Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['val_top_k_categorical_accuracy'])*100, std(history.history['val_top_k_categorical_accuracy'])*100, len(history.history['val_top_k_categorical_accuracy'])))

    return history, model


history, model = train_model(x_train, y_train_ohe, x_val, y_val_ohe)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Categorical Crossentropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig('../../Outputs/model4_DB3_DA_graph_3.png')
plt.close()
model.save('../Models/fungi_classifier_model4_DB3_DA_3.h5')
print("saved")

# evaluation
print("Model evaluation on test dataset: ")
pred_prob = model.predict(x_test)
#pred_class = model.predict_classes(x_test)
pred_class = np.argmax(pred_prob, axis=-1)

# metrics
accuracy = accuracy_score(y_test, pred_class)
k_accuracy = top_k_accuracy_score(y_test, pred_prob, k=2)
print('accuracy =  %.3f' % (accuracy * 100.0),
      'top-2 accuracy = %.3f' % (k_accuracy*100))
"""precision = precision_score(y_test, pred_class)
recall = recall_score(y_test, pred_class)"""
report = classification_report(y_test, pred_class)
print("Classification Report: ")
print(report)
"""f1 = f1_score(y_test, pred_class,average='macro')
print("f1 score: ", f1)"""
confusionMatrix = confusion_matrix(
    y_test, pred_class)  # row(true), column(predicted)
np.set_printoptions(threshold=sys.maxsize)
print("Confusion matrix: ")
print(confusionMatrix)
np.set_printoptions(threshold=False)
#cm_labels = [x for x in range(20)]
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusionMatrix, display_labels=labels)
disp.plot(xticks_rotation=35)
# plt.show()
plt.savefig('../../Outputs/model4_DB3_DA_confusionMatrix_3.png')
plt.show()
plt.close()
# calculate AUC
auc_all = roc_auc_score(y_test, pred_prob, multi_class='ovr')
print('AUC: %.3f' % auc_all)
y_test = np.array(y_test)

# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
for i in range(categories_n):
    y_test_bin = (y_test == i).astype(np.int32)
    y_score = pred_prob[:, i]
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)

    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')

plt.title('ROC curve for fungi classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend(['No skill', 'ROC Curve'], loc='lower right')
plt.savefig('../../Outputs/model4_DB3_DA_roc_3.png')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(categories_n):
    y_test_bin = (y_test == i).astype(np.int32)
    y_score = pred_prob[:, i]
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_display.plot(ax=ax)
#plt.legend(['acc', 'val_acc'], loc='lower right')
#plt.savefig(os.path.join(result_dir, 'roc.png'))
fig.savefig('../../Outputs/model4_DB3_DA_display_roc_3.png')
# show the plot
# plt.show()
plt.close(fig)
