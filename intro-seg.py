#!/usr/bin/env python
# coding: utf-8

# # Introduction to Semantic Segmentation

# ## What is Semantic Segmentation?
# 
# Semantic Segmentation is an image analysis task in which we classify each pixel in the image into a class. <br/>
# 
# Similar to what us humans do all the time by default, when are looking then whatever we are seeing if we think of that as an image
# then we know what class each pixel of the image belongs to.
# 
# Essentially, Semantic Segmentation is the technique through which we can achieve this in Computers.
# 
# There are a few more types of Segmentation, you can read about it more here: https://www.learnopencv.com/image-segmentation/
# This blog will focus on Semantic Segmentation
# 
# So, let's say we have the following image.
# 
# ![](https://lh3.googleusercontent.com/-ELUnFgFJqUU/XPPXOOmhfMI/AAAAAAAAAP0/2cabsTI9uGUYxM3O3w4EOxjR_iJvEQAvACK8BGAs/s374/index3.png)
# <small> Source: Pexels </small>
# 
# And then given the above image its semantically segmentated image would be the following
# 
# ![](https://lh3.googleusercontent.com/-gdUavPeOxdg/XPPXQngAnvI/AAAAAAAAAQA/yoksBterCGQGt-lv3aX4kfyMUDXTar7yACK8BGAs/s374/index4.png)
# 
# As you can see, that each pixel in the image is classified to its respective class.
# 
# This is in most simple terms what Semantic Segmentation is.

# ## Applications of Segmentation
# 
# 
# The most common use case for the Semantic Segmentation is in:
# 
# 1. **Autonomous Driving**
# 
#   <img src="https://cdn-images-1.medium.com/max/1600/1*JKmS08bllQ8SCajIPyiBBQ.png" width="400"/> <br/>
#   <small> Source: CityScapes Dataset </small>
#   
#   In autonomous driving, the image which comes in from the camera is semantically segmented, thus each pixel in the image is classified
#   into a class. This helps the computer understand what is present in the its surroundings and thus helps the car act accordingly.
# 
# 
# 2. **Facial Segmentation**
# 
#   <img src="https://i.ytimg.com/vi/vrvwfFej_r4/maxresdefault.jpg" width="400"/> <br/>
#   <small> Source: https://github.com/massimomauro/FASSEG-repository/blob/master/papers/multiclass_face_segmentation_ICIP2015.pdf </small>
# 
#   Facial Segmentation is used for segmenting each part of the face into a category, like lips, eyes etc. This technique is used for
#   many purposes such as gender estimation, age estimation, facial expression analysis, emotional analysis and more.
#   
# 
# 3. **Indoor Object Segmentation**
# 
#   <img src="https://cs.nyu.edu/~silberman/rmrc2014/header_semantic_segmentation.jpg" width="400"/><br/>
#   <small> Source: http://buildingparser.stanford.edu/dataset.html </small>
# 
#   Guess where is this used? In AR (Augmented Reality) and VR (Virtual Reality). AR applications when required segments the entire indoor area to understand where there 
#   are chairs, tables, people, wall, and other obstacles and so on.
#  
# 
# 4. **Geo-Land Sensing**
# 
#   <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0924271616305305-fx1_lrg.jpg" width="400"/> <br/>
#   <small> Source: https://www.sciencedirect.com/science/article/pii/S0924271616305305 </small>
# 
#   Geo Land Sensing is a way of categorizing each pixel in satellite images into a category such that we can track the land cover of each
#   area. So, say in some area there is a heavy deforestation taking place then appropriate measures can be taken.
# 

# ## Using torchvision for Semantic Segmentation
# 
# Now before we get started, we need to know about the inputs and outputs of these semantic segmentation models.<br/>
# So, let's start!
# 
# These models expect a 3-channled image which is normalized with the Imagenet mean and standard deviation, i.e., <br/>
# `mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]`
# 
# So, the input is `[Ni x Ci x Hi x Wi]`<br/>
# where,
# - `Ni` -> the batch size
# - `Ci` -> the number of channels (which is 3)
# - `Hi` -> the height of the image
# - `Wi` -> the width of the image
# 
# And the output of the model is `[No x Co x Ho x Wo]`<br/>
# where,
# - `No` -> is the batch size (same as `Ni`)
# - `Co` -> **is the number of classes that the dataset have!**
# - `Ho` -> the height of the image (which is the same as `Hi` in almost all cases)
# - `Wo` -> the width of the image (which is the same as `Wi` in almost all cases)
# 
# Alright! And just one more thing!
# The `torchvision` models outputs an `OrderedDict` and not a `torch.Tensor` <br/>
# And in `.eval()` mode it just has one key `out` and thus to get the output we need to get the value
# stored in that `key`.
# 
# The `out` key of this `OrderedDict` is the key that holds the output. <br/>
# So, this `out` key's value has the shape of `[No x Co x Ho x Wo]`.
# 
# Now! we are ready to play :)

# ### FCN with Resnet-101 backbone

# FCN - Fully Convolutional Netowrks, are among the most early invented Neural Networks for the task of Semantic Segmentation.
# 
# Let's load one up!

# In[9]:


from torchvision import models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()


# And that's it we have a pretrained model of `FCN` (which stands for Fully Convolutional Neural Networks) with a `Resnet101` backbone :)
# 
# Now, let's get an image!

# In[10]:


from PIL import Image
import matplotlib.pyplot as plt
import torch

img = Image.open('bird.jpg')
plt.imshow(img); plt.show()


# Now, that we have the image we need to preprocess it and normalize it! <br/>
# So, for the preprocessing steps, we:
# - Resize the image to `(256 x 256)`
# - CenterCrop it to `(224 x 224)`
# - Convert it to Tensor - all the values in the image becomes between `[0, 1]` from `[0, 255]`
# - Normalize it with the Imagenet specific values `mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]`
# 
# And lastly, we unsqueeze the image so that it becomes `[1 x C x H x W]` from `[C x H x W]` <br/>
# We need a batch dimension while passing it to the models.

# In[ ]:


# Apply the transformations needed
import torchvision.transforms as T
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)


# Let's see what the above code cell does </br>
# `T.Compose` is a function that takes in a `list` in which each element is of `transforms` type and </br>
# it returns a object through which we can
# pass batches of images and all the required transforms will be applied to the images.
# 
# Let's take a look at the transforms applied on the images:
# - `T.Resize(256)` : Resizes the image to size `256 x 256`
# - `T.CenterCrop(224)` : Center Crops the image to have a resulting size of `224 x 224`
# - `T.ToTensor()` : Converts the image to type `torch.Tensor` and have values between `[0, 1]`
# - `T.Normalize(mean, std)` : Normalizes the image with the given mean and standard deviation.
# 
# Alright! Now that we have the image all preprocessed and ready! Let's pass it through the model and get the `out` key.<br/>
# As I said, the output of the model is a `OrderedDict` so, we need to take the `out` key from that to get the output of the model.

# In[12]:


# Pass the input through the net
out = fcn(inp)['out']
print (out.shape)


# Alright! So, `out` is the final output of the model. And as we can see, its shape is `[1 x 21 x H x W]` as discussed earlier. So, the model was trained on `21` classes and thus our output have `21` channels!<br/>
# 
# Now, what we need to do is make this `21` channeled output into a `2D` image or a `1` channeled image, where each pixel of that image corresponds to a class!
# 
# So, the `2D` image, (of shape `[H x W]`) will have each pixel corresponding to a class label, and thus <br/>
# for each `(x, y)` in this `2D` image will correspond to a number between `0 - 20` representing a class.
# 
# And how do we get there from this `[1 x 21 x H x W]`?<br/>
# We take a max index for each pixel position, which represents the class<br/>

# In[13]:


import numpy as np
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (om.shape)
print (np.unique(om))


# Alright! So, we as we can see now have a `2D` image. Where each pixel corresponds to a class!
# The last thing is to take this `2D` image where each pixel corresponds to a class label and convert this<br/>
# into a segmentation map where each class label is converted into a `RGB` color and thus helping in easy visualization.
# 
# We will use the following function to convert this `2D` image to an `RGB` image wheree each label is mapped to its
# corresponding color.

# In[ ]:


# Define the helper function
def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb


# Let's see what we are doing inside this function!
# 
# first `label_colors` stores the colors for each of the clases, according to the index </br>
# So, the color for the  first class which is `background` is stored in the `0`th index of the `label_colors` list, 
# the second class which is `aeroplane` is stored at index `1` of `label_colors`.
# 
# Now, we are to create an `RGB` image from the `2D` image passed. So, what we do, is we create empty `2D` matrices for all 3 channels.
# 
# So, `r`, `g`, and `b` are arrays which will form the `RGB` channels for the final image. And each are of shape `[H x W]` 
# (which is same as the shape of `image` passed in)
# 
# Now, we loop over each class color we stored in `label_colors`.
# And we get the indexes in the image where that particular class label is present. (`idx = image == l`)
# And then for each channel, we put its corresponding color to those pixels where that class label is present.
# 
# And finally we stack the 3 seperate channels to form a `RGB` image.
# 
# Okay! Now, let's use this function to see the final segmented output!

# In[ ]:


rgb = decode_segmap(om)
plt.imshow(rgb); plt.show()


# And there we go!!<br/>
# Wooohooo! We have segmented the output of the image. 
# 
# That's the bird!
# 
# Also, Do note that the image after segmentation is smaller than the original image as in the preprocessing step the image is resized and cropped.
# 
# Next, let's move all this under one function and play with a few more images!

# In[ ]:


def segment(net, path, show_orig=True, dev='cuda'):
  img = Image.open(path)
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = T.Compose([T.Resize(640), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  rgb = decode_segmap(om)
  plt.imshow(rgb); plt.axis('off'); plt.show()


# And let's get a new image!

# In[ ]:




# ### DeepLabv3

# In[16]:


dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()



import time

def infer_time(net, path='bird.jpg', dev='cuda'):
  img = Image.open(path)
  trf = T.Compose([T.Resize(256), 
                   T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  
  inp = trf(img).unsqueeze(0).to(dev)
  
  st = time.time()
  out1 = net.to(dev)(inp)
  et = time.time()
  
  return et - st


# **On CPU**

# In[ ]:


avg_over = 100

fcn_infer_time_list_cpu = [infer_time(fcn, dev='cpu') for _ in range(avg_over)]
fcn_infer_time_avg_cpu = sum(fcn_infer_time_list_cpu) / avg_over

dlab_infer_time_list_cpu = [infer_time(dlab, dev='cpu') for _ in range(avg_over)]
dlab_infer_time_avg_cpu = sum(dlab_infer_time_list_cpu) / avg_over


print ('Inference time for first few calls for FCN      : {}'.format(fcn_infer_time_list_cpu[:10]))
print ('Inference time for first few calls for DeepLabv3: {}'.format(dlab_infer_time_list_cpu[:10]))

print ('The Average Inference time on FCN is:     {:.2f}s'.format(fcn_infer_time_avg_cpu))
print ('The Average Inference time on DeepLab is: {:.2f}s'.format(dlab_infer_time_avg_cpu))


# **On GPU**

# In[ ]:


avg_over = 100

fcn_infer_time_list_gpu = [infer_time(fcn) for _ in range(avg_over)]
fcn_infer_time_avg_gpu = sum(fcn_infer_time_list_gpu) / avg_over

dlab_infer_time_list_gpu = [infer_time(dlab) for _ in range(avg_over)]
dlab_infer_time_avg_gpu = sum(dlab_infer_time_list_gpu) / avg_over

print ('Inference time for first few calls for FCN      : {}'.format(fcn_infer_time_list_gpu[:10]))
print ('Inference time for first few calls for DeepLabv3: {}'.format(dlab_infer_time_list_gpu[:10]))

print ('The Average Inference time on FCN is:     {:.3f}s'.format(fcn_infer_time_avg_gpu))
print ('The Average Inference time on DeepLab is: {:.3f}s'.format(dlab_infer_time_avg_gpu))


# 
# We can see that in both cases (for GPU and CPU) its taking longer for the DeepLabv3 model, as its a much deeper model as compared to FCN.
# 
# Also, we have printed out the first few inference times for each model. Something we can notice is that the inference time for the first call
# takes quite long than the others . This is because after the 1st call a lot of the calculations required are cached and thus its faster for the next calls.
# 
# Nice! Now, let's try to vizualize the difference in the time taken for the CPU and the GPU.

# In[ ]:


plt.bar([0.1, 0.2], [fcn_infer_time_avg_cpu, dlab_infer_time_avg_cpu], width=0.08)
plt.ylabel('Time taken in Seconds')
plt.xticks([0.1, 0.2], ['FCN', 'DeepLabv3'])
plt.title('Inference time of FCN and DeepLabv3 on CPU')
plt.show()


# In[ ]:


plt.bar([0.1, 0.2], [fcn_infer_time_avg_gpu, dlab_infer_time_avg_gpu], width=0.08)
plt.ylabel('Time taken in Seconds')
plt.xticks([0.1, 0.2], ['FCN', 'DeepLabv3'])
plt.title('Inference time of FCN and DeepLabv3 on GPU')
plt.show()


# Okay! Now, let's move on to the next comparison, where we will compare the model sizes for both the models.

# ### Model Size

# In[17]:


import os

resnet101_size = os.path.getsize('/root/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth')
fcn_size = os.path.getsize('/root/.cache/torch/checkpoints/fcn_resnet101_coco-7ecb50ca.pth')
dlab_size = os.path.getsize('/root/.cache/torch/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth')

fcn_total = fcn_size + resnet101_size
dlab_total = dlab_size + resnet101_size
    
print ('Size of the FCN model with Resnet101 backbone is:       {:.2f} MB'.format(fcn_total /  (1024 * 1024)))
print ('Size of the DeepLabv3 model with Resnet101 backbone is: {:.2f} MB'.format(dlab_total / (1024 * 1024)))


# In[27]:


plt.bar([0, 1], [fcn_total / (1024 * 1024), dlab_total / (1024 * 1024)])
plt.ylabel('Size of the model in MegaBytes')
plt.xticks([0, 1], ['FCN', 'DeepLabv3'])
plt.title('Comparison of the model size of FCN and DeepLabv3')
plt.show()


# ## Conclusion

# Hope you enjoyed this tutorial!
# 
# Feel free to leave comments and any feedback you wish! If you would like to learn<br/>
# more about this, how these techniques work and how to implement these models!
# 
# Please do check out the `Deep Learning with PyTorch` Course from OpenCV.org! <br/>
# Link: https://opencv.org/ai-courses-by-opencv-kickstarter-campaign/

# In[ ]:




