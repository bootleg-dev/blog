[//]: # (---)

[//]: # (title: "Face Landmarks Detection using CNN")

[//]: # (description: "Can computers really understand the human face?")

[//]: # (dateString: "Date: 01 June, 2024")

[//]: # (date: "2024-06-01T03:55:49.337541+05:00")

[//]: # (draft: false)

[//]: # (tags: ["DL", "AI", "Python", "PyTorch"])

[//]: # (weight: 1)

[//]: # (cover:)

[//]: # (    image: "/posts/face-landmarks-detection/cover.jpg")

[//]: # (    # caption: "A sample landmark detection on a photo by Ayo Ogunseinde taken from Unsplash")

[//]: # (---)

[//]: # ()
[//]: # (# Introduction)

[//]: # ()
[//]: # (Ever wondered how Instagram applies stunning filters to your face? The software detects key points on your face and projects a mask on top. This tutorial will guide you on how to build one such software using Pytorch.)

[//]: # ()
[//]: # (# Dataset)

[//]: # ()
[//]: # (In this tutorial, we will use the official [DLib Dataset]&#40;http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz&#41; which contains **6666 images of varying dimensions**. Additionally, *labels_ibug_300W_train.xml* &#40;comes with the dataset&#41; contains the coordinates of **68 landmarks for each face**. The script below will download the dataset and unzip it in Colab Notebook.)

[//]: # ()
[//]: # (```python)

[//]: # (if not os.path.exists&#40;'/content/ibug_300W_large_face_landmark_dataset'&#41;:)

[//]: # (    !wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)

[//]: # (    !tar -xvzf 'ibug_300W_large_face_landmark_dataset.tar.gz'    )

[//]: # (    !rm -r 'ibug_300W_large_face_landmark_dataset.tar.gz')

[//]: # (```)

[//]: # ()
[//]: # (Here is a sample image from the dataset. We can see that the face occupies a very small fraction of the entire image. If we feed the full image to the neural network, it will also process the background &#40;irrelevant information&#41;, making it difficult for the model to learn. Therefore, we need to crop the image and feed only the face portion.)

[//]: # ()
[//]: # (![Sample Image and Landmarks from the Dataset]&#40;/blog/face-landmarks-detection/img1.jpg&#41;)

[//]: # ()
[//]: # (## Data Preprocessing)

[//]: # ()
[//]: # (To prevent the neural network from overfitting the training dataset, we need to randomly transform the dataset. We will apply the following operations to the training and validation dataset:)

[//]: # ()
[//]: # (- Since the face occupies a very small portion of the entire image, crop the image and use only the face for training.)

[//]: # (- Resize the cropped face into a &#40;224x224&#41; image.)

[//]: # (- Randomly change the brightness and saturation of the resized face.)

[//]: # (- Randomly rotate the face after the above three transformations.)

[//]: # (- Convert the image and landmarks into torch tensors and normalize them between [-1, 1].)

[//]: # ()
[//]: # (```python)

[//]: # (class Transforms&#40;&#41;:)

[//]: # (    def __init__&#40;self&#41;:)

[//]: # (        pass)

[//]: # (    )
[//]: # (    def rotate&#40;self, image, landmarks, angle&#41;:)

[//]: # (        angle = random.uniform&#40;-angle, +angle&#41;)

[//]: # ()
[//]: # (        transformation_matrix = torch.tensor&#40;[)

[//]: # (            [+cos&#40;radians&#40;angle&#41;&#41;, -sin&#40;radians&#40;angle&#41;&#41;], )

[//]: # (            [+sin&#40;radians&#40;angle&#41;&#41;, +cos&#40;radians&#40;angle&#41;&#41;])

[//]: # (        ]&#41;)

[//]: # ()
[//]: # (        image = imutils.rotate&#40;np.array&#40;image&#41;, angle&#41;)

[//]: # ()
[//]: # (        landmarks = landmarks - 0.5)

[//]: # (        new_landmarks = np.matmul&#40;landmarks, transformation_matrix&#41;)

[//]: # (        new_landmarks = new_landmarks + 0.5)

[//]: # (        return Image.fromarray&#40;image&#41;, new_landmarks)

[//]: # ()
[//]: # (    def resize&#40;self, image, landmarks, img_size&#41;:)

[//]: # (        image = TF.resize&#40;image, img_size&#41;)

[//]: # (        return image, landmarks)

[//]: # ()
[//]: # (    def color_jitter&#40;self, image, landmarks&#41;:)

[//]: # (        color_jitter = transforms.ColorJitter&#40;brightness=0.3, )

[//]: # (                                              contrast=0.3,)

[//]: # (                                              saturation=0.3, )

[//]: # (                                              hue=0.1&#41;)

[//]: # (        image = color_jitter&#40;image&#41;)

[//]: # (        return image, landmarks)

[//]: # ()
[//]: # (    def crop_face&#40;self, image, landmarks, crops&#41;:)

[//]: # (        left = int&#40;crops['left']&#41;)

[//]: # (        top = int&#40;crops['top']&#41;)

[//]: # (        width = int&#40;crops['width']&#41;)

[//]: # (        height = int&#40;crops['height']&#41;)

[//]: # ()
[//]: # (        image = TF.crop&#40;image, top, left, height, width&#41;)

[//]: # ()
[//]: # (        img_shape = np.array&#40;image&#41;.shape)

[//]: # (        landmarks = torch.tensor&#40;landmarks&#41; - torch.tensor&#40;[[left, top]]&#41;)

[//]: # (        landmarks = landmarks / torch.tensor&#40;[img_shape[1], img_shape[0]]&#41;)

[//]: # (        return image, landmarks)

[//]: # ()
[//]: # (    def __call__&#40;self, image, landmarks, crops&#41;:)

[//]: # (        image = Image.fromarray&#40;image&#41;)

[//]: # (        image, landmarks = self.crop_face&#40;image, landmarks, crops&#41;)

[//]: # (        image, landmarks = self.resize&#40;image, landmarks, &#40;224, 224&#41;&#41;)

[//]: # (        image, landmarks = self.color_jitter&#40;image, landmarks&#41;)

[//]: # (        image, landmarks = self.rotate&#40;image, landmarks, angle=10&#41;)

[//]: # (        )
[//]: # (        image = TF.to_tensor&#40;image&#41;)

[//]: # (        image = TF.normalize&#40;image, [0.5], [0.5]&#41;)

[//]: # (        return image, landmarks)

[//]: # (```)

[//]: # ()
[//]: # (# Dataset Class)

[//]: # ()
[//]: # (Now that we have our transformations ready, let’s write our dataset class. The *labels_ibug_300W_train.xml* contains the image path, landmarks and coordinates for the bounding box &#40;for cropping the face&#41;. We will store these values in lists to access them easily during training. In this tutorial, the neural network will be trained on grayscale images.)

[//]: # ()
[//]: # (```python)

[//]: # (class FaceLandmarksDataset&#40;Dataset&#41;:)

[//]: # ()
[//]: # (    def __init__&#40;self, transform=None&#41;:)

[//]: # ()
[//]: # (        tree = ET.parse&#40;'ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml'&#41;)

[//]: # (        root = tree.getroot&#40;&#41;)

[//]: # ()
[//]: # (        self.image_filenames = [])

[//]: # (        self.landmarks = [])

[//]: # (        self.crops = [])

[//]: # (        self.transform = transform)

[//]: # (        self.root_dir = 'ibug_300W_large_face_landmark_dataset')

[//]: # (        )
[//]: # (        for filename in root[2]:)

[//]: # (            self.image_filenames.append&#40;os.path.join&#40;self.root_dir, filename.attrib['file']&#41;&#41;)

[//]: # ()
[//]: # (            self.crops.append&#40;filename[0].attrib&#41;)

[//]: # ()
[//]: # (            landmark = [])

[//]: # (            for num in range&#40;68&#41;:)

[//]: # (                x_coordinate = int&#40;filename[0][num].attrib['x']&#41;)

[//]: # (                y_coordinate = int&#40;filename[0][num].attrib['y']&#41;)

[//]: # (                landmark.append&#40;[x_coordinate, y_coordinate]&#41;)

[//]: # (            self.landmarks.append&#40;landmark&#41;)

[//]: # ()
[//]: # (        self.landmarks = np.array&#40;self.landmarks&#41;.astype&#40;'float32'&#41;     )

[//]: # ()
[//]: # (        assert len&#40;self.image_filenames&#41; == len&#40;self.landmarks&#41;)

[//]: # ()
[//]: # (    def __len__&#40;self&#41;:)

[//]: # (        return len&#40;self.image_filenames&#41;)

[//]: # ()
[//]: # (    def __getitem__&#40;self, index&#41;:)

[//]: # (        image = cv2.imread&#40;self.image_filenames[index], 0&#41;)

[//]: # (        landmarks = self.landmarks[index])

[//]: # (        )
[//]: # (        if self.transform:)

[//]: # (            image, landmarks = self.transform&#40;image, landmarks, self.crops[index]&#41;)

[//]: # ()
[//]: # (        landmarks = landmarks - 0.5)

[//]: # ()
[//]: # (        return image, landmarks)

[//]: # ()
[//]: # (dataset = FaceLandmarksDataset&#40;Transforms&#40;&#41;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (**Note:** `landmarks = landmarks - 0.5` is done to zero-centre the landmarks as zero-centred outputs are easier for the neural network to learn.)

[//]: # ()
[//]: # (The output of the dataset after preprocessing will look something like this &#40;landmarks have been plotted on the image&#41;.)

[//]: # ()
[//]: # (![Preprocessed Data Sample]&#40;/blog/face-landmarks-detection/img2.jpg&#41;)

[//]: # ()
[//]: # (# Neural Network)

[//]: # ()
[//]: # (We will use the ResNet18 as the basic framework. We need to modify the first and last layers to suit our purpose. In the first layer, we will make the input channel count as 1 for the neural network to accept grayscale images. Similarly, in the final layer, the output channel count should equal **68 * 2 = 136** for the model to predict the &#40;x, y&#41; coordinates of the 68 landmarks for each face.)

[//]: # ()
[//]: # (```python)

[//]: # (class Network&#40;nn.Module&#41;:)

[//]: # (    def __init__&#40;self,num_classes=136&#41;:)

[//]: # (        super&#40;&#41;.__init__&#40;&#41;)

[//]: # (        self.model_name='resnet18')

[//]: # (        self.model=models.resnet18&#40;&#41;)

[//]: # (        self.model.conv1=nn.Conv2d&#40;1, 64, kernel_size=7, stride=2, padding=3, bias=False&#41;)

[//]: # (        self.model.fc=nn.Linear&#40;self.model.fc.in_features, num_classes&#41;)

[//]: # (        )
[//]: # (    def forward&#40;self, x&#41;:)

[//]: # (        x=self.model&#40;x&#41;)

[//]: # (        return x)

[//]: # (```)

[//]: # ()
[//]: # (# Training the Neural Network)

[//]: # ()
[//]: # (We will use the Mean Squared Error between the predicted landmarks and the true landmarks as the loss function. Keep in mind that the learning rate should be kept low to avoid exploding gradients. The network weights will be saved whenever the validation loss reaches a new minimum value. Train for at least 20 epochs to get the best performance.)

[//]: # ()
[//]: # (```python)

[//]: # (network = Network&#40;&#41;)

[//]: # (network.cuda&#40;&#41;    )

[//]: # ()
[//]: # (criterion = nn.MSELoss&#40;&#41;)

[//]: # (optimizer = optim.Adam&#40;network.parameters&#40;&#41;, lr=0.0001&#41;)

[//]: # ()
[//]: # (loss_min = np.inf)

[//]: # (num_epochs = 10)

[//]: # ()
[//]: # (start_time = time.time&#40;&#41;)

[//]: # (for epoch in range&#40;1,num_epochs+1&#41;:)

[//]: # (    )
[//]: # (    loss_train = 0)

[//]: # (    loss_valid = 0)

[//]: # (    running_loss = 0)

[//]: # (    )
[//]: # (    network.train&#40;&#41;)

[//]: # (    for step in range&#40;1,len&#40;train_loader&#41;+1&#41;:)

[//]: # (    )
[//]: # (        images, landmarks = next&#40;iter&#40;train_loader&#41;&#41;)

[//]: # (        )
[//]: # (        images = images.cuda&#40;&#41;)

[//]: # (        landmarks = landmarks.view&#40;landmarks.size&#40;0&#41;,-1&#41;.cuda&#40;&#41; )

[//]: # (        )
[//]: # (        predictions = network&#40;images&#41;)

[//]: # (        )
[//]: # (        # clear all the gradients before calculating them)

[//]: # (        optimizer.zero_grad&#40;&#41;)

[//]: # (        )
[//]: # (        # find the loss for the current step)

[//]: # (        loss_train_step = criterion&#40;predictions, landmarks&#41;)

[//]: # (        )
[//]: # (        # calculate the gradients)

[//]: # (        loss_train_step.backward&#40;&#41;)

[//]: # (        )
[//]: # (        # update the parameters)

[//]: # (        optimizer.step&#40;&#41;)

[//]: # (        )
[//]: # (        loss_train += loss_train_step.item&#40;&#41;)

[//]: # (        running_loss = loss_train/step)

[//]: # (        )
[//]: # (        print_overwrite&#40;step, len&#40;train_loader&#41;, running_loss, 'train'&#41;)

[//]: # (        )
[//]: # (    network.eval&#40;&#41; )

[//]: # (    with torch.no_grad&#40;&#41;:)

[//]: # (        )
[//]: # (        for step in range&#40;1,len&#40;valid_loader&#41;+1&#41;:)

[//]: # (            )
[//]: # (            images, landmarks = next&#40;iter&#40;valid_loader&#41;&#41;)

[//]: # (        )
[//]: # (            images = images.cuda&#40;&#41;)

[//]: # (            landmarks = landmarks.view&#40;landmarks.size&#40;0&#41;,-1&#41;.cuda&#40;&#41;)

[//]: # (        )
[//]: # (            predictions = network&#40;images&#41;)

[//]: # ()
[//]: # (            # find the loss for the current step)

[//]: # (            loss_valid_step = criterion&#40;predictions, landmarks&#41;)

[//]: # ()
[//]: # (            loss_valid += loss_valid_step.item&#40;&#41;)

[//]: # (            running_loss = loss_valid/step)

[//]: # ()
[//]: # (            print_overwrite&#40;step, len&#40;valid_loader&#41;, running_loss, 'valid'&#41;)

[//]: # (    )
[//]: # (    loss_train /= len&#40;train_loader&#41;)

[//]: # (    loss_valid /= len&#40;valid_loader&#41;)

[//]: # (    )
[//]: # (    print&#40;'\n--------------------------------------------------'&#41;)

[//]: # (    print&#40;'Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format&#40;epoch, loss_train, loss_valid&#41;&#41;)

[//]: # (    print&#40;'--------------------------------------------------'&#41;)

[//]: # (    )
[//]: # (    if loss_valid < loss_min:)

[//]: # (        loss_min = loss_valid)

[//]: # (        torch.save&#40;network.state_dict&#40;&#41;, '/content/face_landmarks.pth'&#41; )

[//]: # (        print&#40;"\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format&#40;loss_min, epoch, num_epochs&#41;&#41;)

[//]: # (        print&#40;'Model Saved\n'&#41;)

[//]: # (     )
[//]: # (print&#40;'Training Complete'&#41;)

[//]: # (print&#40;"Total Elapsed Time : {} s".format&#40;time.time&#40;&#41;-start_time&#41;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (# Predict on Unseen Data)

[//]: # ()
[//]: # (Use the code snippet below to predict landmarks in unseen images.)

[//]: # ()
[//]: # (```python)

[//]: # (import time)

[//]: # (import cv2)

[//]: # (import os)

[//]: # (import numpy as np)

[//]: # (import matplotlib.pyplot as plt)

[//]: # (from PIL import Image)

[//]: # (import imutils)

[//]: # ()
[//]: # (import torch)

[//]: # (import torch.nn as nn)

[//]: # (from torchvision import models)

[//]: # (import torchvision.transforms.functional as TF)

[//]: # (#######################################################################)

[//]: # (image_path = 'pic.jpg')

[//]: # (weights_path = 'face_landmarks.pth')

[//]: # (frontal_face_cascade_path = 'haarcascade_frontalface_default.xml')

[//]: # (#######################################################################)

[//]: # (class Network&#40;nn.Module&#41;:)

[//]: # (    def __init__&#40;self,num_classes=136&#41;:)

[//]: # (        super&#40;&#41;.__init__&#40;&#41;)

[//]: # (        self.model_name='resnet18')

[//]: # (        self.model=models.resnet18&#40;pretrained=False&#41;)

[//]: # (        self.model.conv1=nn.Conv2d&#40;1, 64, kernel_size=7, stride=2, padding=3, bias=False&#41;)

[//]: # (        self.model.fc=nn.Linear&#40;self.model.fc.in_features,num_classes&#41;)

[//]: # (        )
[//]: # (    def forward&#40;self, x&#41;:)

[//]: # (        x=self.model&#40;x&#41;)

[//]: # (        return x)

[//]: # ()
[//]: # (#######################################################################)

[//]: # (face_cascade = cv2.CascadeClassifier&#40;frontal_face_cascade_path&#41;)

[//]: # ()
[//]: # (best_network = Network&#40;&#41;)

[//]: # (best_network.load_state_dict&#40;torch.load&#40;weights_path, map_location=torch.device&#40;'cpu'&#41;&#41;&#41; )

[//]: # (best_network.eval&#40;&#41;)

[//]: # ()
[//]: # (image = cv2.imread&#40;image_path&#41;)

[//]: # (grayscale_image = cv2.cvtColor&#40;image, cv2.COLOR_BGR2GRAY&#41;)

[//]: # (display_image = cv2.cvtColor&#40;image, cv2.COLOR_BGR2RGB&#41;)

[//]: # (height, width,_ = image.shape)

[//]: # ()
[//]: # (faces = face_cascade.detectMultiScale&#40;grayscale_image, 1.1, 4&#41;)

[//]: # ()
[//]: # (all_landmarks = [])

[//]: # (for &#40;x, y, w, h&#41; in faces:)

[//]: # (    image = grayscale_image[y:y+h, x:x+w])

[//]: # (    image = TF.resize&#40;Image.fromarray&#40;image&#41;, size=&#40;224, 224&#41;&#41;)

[//]: # (    image = TF.to_tensor&#40;image&#41;)

[//]: # (    image = TF.normalize&#40;image, [0.5], [0.5]&#41;)

[//]: # ()
[//]: # (    with torch.no_grad&#40;&#41;:)

[//]: # (        landmarks = best_network&#40;image.unsqueeze&#40;0&#41;&#41; )

[//]: # ()
[//]: # (    landmarks = &#40;landmarks.view&#40;68,2&#41;.detach&#40;&#41;.numpy&#40;&#41; + 0.5&#41; * np.array&#40;[[w, h]]&#41; + np.array&#40;[[x, y]]&#41;)

[//]: # (    all_landmarks.append&#40;landmarks&#41;)

[//]: # ()
[//]: # (plt.figure&#40;&#41;)

[//]: # (plt.imshow&#40;display_image&#41;)

[//]: # (for landmarks in all_landmarks:)

[//]: # (    plt.scatter&#40;landmarks[:,0], landmarks[:,1], c = 'c', s = 5&#41;)

[//]: # ()
[//]: # (plt.show&#40;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (> ⚠️ The above code snippet will not work in Colab Notebook as some functionality of the OpenCV is not supported in Colab yet. To run the above cell, use your local machine.)

[//]: # ()
[//]: # (**OpenCV Harr Cascade Classifier** is used to detect faces in an image. Object detection using Haar Cascades is a machine learning-based approach where a cascade function is trained with a set of input data. OpenCV already contains many pre-trained classifiers for face, eyes, pedestrians, and many more. In our case, we will be using the face classifier for which you need to download the pre-trained classifier XML file and save it to your working directory.)

[//]: # ()
[//]: # (![Face Detection]&#40;/blog/face-landmarks-detection/img3.jpg&#41;)

[//]: # ()
[//]: # (Detected faces in the input image are then cropped, resized to **&#40;224, 224&#41;** and fed to our trained neural network to predict landmarks in them.)

[//]: # ()
[//]: # (![Landmarks Detection on the Cropped Face ]&#40;/blog/face-landmarks-detection/img4.jpg&#41;)

[//]: # ()
[//]: # (The predicted landmarks in the cropped faces are then overlayed on top of the original image. The result is the image shown below. Pretty impressive, right!)

[//]: # ()
[//]: # (![Final Result]&#40;/blog/face-landmarks-detection/cover.jpg&#41;)

[//]: # ()
[//]: # (Similarly, landmarks detection on multiple faces:)

[//]: # ()
[//]: # (![Detection on multiple faces]&#40;/blog/face-landmarks-detection/img5.jpg&#41;)

[//]: # ()
[//]: # (Here, you can see that the OpenCV Harr Cascade Classifier has detected multiple faces including a false positive &#40;a fist is predicted as a face&#41;. So, the network has plotted some landmarks on that.)

[//]: # ()
[//]: # (# That’s all folks!)

[//]: # (If you made it till here, hats off to you! You just trained your very own neural network to detect face landmarks in any image. Try predicting face landmarks on your webcam feed!!)

[//]: # ()
[//]: # (# Colab Notebook)

[//]: # (The complete code can be found in the interactive [Colab Notebook]&#40;https://colab.research.google.com/drive/1TOw7W_WU4oltoGZfZ_0krpxmhdFR2gmb&#41;.)