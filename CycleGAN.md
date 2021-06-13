# Reproduction and evaluation of loss functions on CycleGan by Anish Sridharan and Agrim Sharma
                                                         
    
Cyclegan is a model which converts a dataset from on style to another and vice-versa, without paired images. That is it is used to convert a collection of images of one type to the target type, with both the datasets having no relationship. Some examples of these are the popular horse to zebra conversion, or converting photographs to artisic styles such as monet, it us also used in sim-to-real with transforming cityscape images to real-life, and winter to summer etc. In this blog we will be discussing the architcture and implementation of the CycleGan and the effect of varying the loss function in the image translation. To show and discuss the results, we use the apple to orange dataset, that is converting the images of apple to oranges, and vice-versa.

## How it works
### CycleGan Architecture
In order to achive the translation the CycleGan architecture has two GANS. that is, two pairs of Generator and Discriminator. One for translating Oranges to Apples, and other for apple to oranges.
The generators take an image as the input, instead of noise (like usual GANs) and outputs an image  where an apple/orange is converted, while rest of it remains the same. The discriminators are trained to identify if the images are real or fake, thus helping the generators, produce better images. 

Cycle GAN uses PatchGan70, where both generator take an input of size 256x256. This is then downsampled and then is upsampled to create the new generated image at the same size as the input. The discriminator in this structure, takes the generated image as the input. It then goes thorugh layers of convolution, extracting features and then classifies 70x70 overlapping patches as real or fake in a 30x30 output tensor. This output is then averaged out and the classification results for the overall image is found.<center>![](https://i.imgur.com/HXrZ6wf.png =600x)</center>

Adam optimizer is used and the original model has a learning rate of 2e-4 for the first 100 epochs, and then linearly increased to 2e-6 during the last 100.
#### GAN Loss
The GAN loss function for training is as follows:<center>![](https://i.imgur.com/Zwcs6QZ.png =500x)</center>

This formula, also known as the adversarial loss, allows the GAN to produce better images. The Generator tries to produce a fake image G(x) and the Discriminator Dy tries to distinguish between real or fake images. By maximizing the loss with respect to the discriminator, and minimizing the loss with respect to the generator, we are able to produce a GAN that generates images indistinguishable from real images. The CycleGAN computes the adversarial loss for both the GANs this way, and is added to the final Loss.
#### Cycle Consistency Loss
In addition to the GAN model and losses, CycleGan also has the cycle consistency loss, also called as the reconstruction loss. This is used so that mode collapse does not happen. In image to image translation, mode collapse occurs when, all the translations from one of the dataset to the same output image, and hence we do not get meaningful translations. <center>![](https://i.imgur.com/h2u1W2A.png =450x)</center>

The cycle consistency loss is the L1 norm, used to find out the diviation between the original image and the reconstruction of the original image from the fake image. This loss has both forward and backward losses, that is it is done for both the generators. <center>![](https://i.imgur.com/mhTuXGm.png =600x)</center>
#### Identity Loss
This Loss function ensures that the GAN only performs changes to the target object. Let G_A be the generator that translates from the dataset O(Orange) to A(Apples) and G_B translat from A to O. D_A and D_B are the respective discriminators. To calculate this loss, G_A is input an image of an Apple, and therefore, the resulting image should also be the same apple. Similarly, G_B when input an image of an orange should produce the same image of an orange. Additionally, real_A and real_B represent the real images, rec_A and rec_B are the images recreated by passing them through both the GANs. And fake_A and fake_B are translated images.

The overall CycleGan Loss is the weighted sum of the adversarial, identity and the cycle consitancy loss where all the losses can be weighted by a term lambda to emphasize or subdue their effect. The pseudo code for this is as follows:
```
criterionGAN = networks.GANLoss()
criterionCycle = torch.nn.L1Loss()
criterionIdt = torch.nn.L1Loss()
            
idt_A = netG_A(real_B)
idt_B = netG_B(real_A)
loss_idt_A = criterionIdt(idt_A, real_B) * lambda * lambda_idt
loss_idt_B = criterionIdt(idt_B, real_A) * lambda * lambda_idt

loss_G_A = criterionGAN(netD_A(fake_B), True)
loss_G_B = criterionGAN(netD_B(fake_A), True)

loss_cycle_A = criterionCycle(rec_A, real_A) * lambda
loss_cycle_B = criterionCycle(rec_B, real_B) * lambda

loss_G = loss_G_A + loss_G_B 
       + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
```


In this blog we will not only talk about the results generated by the original CycleGan itself, but also discuss two other loss functions aiming to create better mappings by 1) Improving the feature consistance 2) Better shape translations
## Additions to the CycleGan
### Feature level cycle consistency
The original cycle consistency consistency loss tries to minimize the difference between the pixels of the real image and the reconstructed image. While this proves to be a viable way of training the GAN,the authors of CycleGAN with Better Cycles suggest that in a practical setting, we do not necessarily need pixel level precision to differentiate between two distinct objects. A GAN that also captures feature level differences can also produce meaningful results even though it has a "weaker" cycle consistency requirement because the resulting image need not be exactly the same as the real image, as long as it has a reasonable amount of resemblance. 

In order to achieve this, we added a hyper-parameter called gamma, which starts off small and increases over epochs. The loss function is also changed as shown below:<center>![](https://i.imgur.com/KPXqHpO.png =600x)</center>
Here, the f_Dx represents the discriminator network, without its last layer that is used for classification into real or fake. This acts as the feature extractor that evolves over time as the discriminator gets better at differentiating between real and fake images. We take the features of the reconstructed image, and the features of the real image, and use the the L1 loss between them to depict a feature level loss. This is weighted by gamma which is small initially because the discriminators are not good at extracting features at the start of training. This gamma weighted L1 loss is added to the original cycle consistency loss, which is now weighted by (1 - gamma) to compensate for the new addition. 

From the formula, we can see that initially, the original cycle consistency loss dominates the loss function, focusing on pixel level accuracy. Over time, as the GAN learns the pixel level representations, the gamma value increases, and the focus shifts to feature level accuracy. This allows the GAN to generalize better, resulting in the generators being able to create accurate fakes even with a greater variance in the dataset. This implementation can be seen in the pseudo code below:
```
discriminator_loss_recA = netD_A.module.model[:-1](rec_A)
discriminator_loss_realA = self.netD_A.module.model[:-1](real_A)
loss_cycle_A = (criterionCycle(discriminator_loss_recA, discriminator_loss_realA) * gamma 
              + criterionCycle(rec_A, real_A) * (1 - gamma)) * lambda
              
discriminator_loss_recB  = netD_B.module.model[:-1](rec_B)
discriminator_loss_realB = netD_B.module.model[:-1](real_B)
loss_cycle_B = (criterionCycle(discriminator_loss_recB, discriminator_loss_realB) * gamma
              + criterionCycle(rec_B, real_B) * (1 - gamma)) * lambda
              
loss_G = loss_G_A + loss_G_B 
       + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
```
Gamma was initialized at 0, and was increased linearly till 0.9 every epoch until epoch 100.

### Cycle Consistency Weight Decay
The authors of CycleGAN with Better Cycles, hypothesised that, cycle-consistency helps stabilize CycleGan in the early stages, but a gradual decay of the cycle consistency would help the model to create a more realistic image as the training goes on. We tested this hypothesis as a standalone model, and also combined this with their other suggestion of the feature level cycle consistency as explained above.
### Embedded CycleGan
The authors in this work tried to modify the existing CycleGan architecture to make it generate better shappe translations and better reconstructions. In order to achieve this he did hypothesis the reasons why the original architecture did not capture the shape changes. They are:
 1) Cycle-Consistency loss is too significant and dominates the generator loss- The cycle consistency loss is used so that model better keeps the identity of the original image. But while doing this, the amound by which shape of the original item can be modified, is restricted.
 2) Loss of information - For the reconstruction of the original image, when the fake image is used as input, it should contain information required for the reconstruction. But when there are shape changes between the items (such as apple and oranges), there might be a situation where the fake image does not contain enough information for accurate reconstructions.
 3) Weak Discriminator-  If the discriminator cannot accurately learn the features of the class, there will be problems in accurately translating images to the class.
 4) Inconsistent dataset or insufficient data: This is self explanatory, there needs to be enough and consistent data for the accurate feature learning and translation.

The work of the author touches on the first two points, that is use lower cycle-consistency loss, while also alowing for better reconstruction.
The Embedded CycleGan is a minor modification to the architecture of the existing CycleGan where an embedding channel is added to the 3 channel RGB input image. This extra channel can be used by the Generator to encode information such as shape. When adding the embedding channel as the input to the generator, it is initialized with just a tensor of zeros. The Generator then performs its computations, resulting in a 4 channel output image. The embedding channel can then be removed from the generated image and we take the L1 loss between the generated embedding channel and the input embedding channel. This loss is also added to the final loss in the backward function. The idea behind the 4th channel is to allow to the generator to learn information about shapes, which it could then possibly use to change the shape of the input image (example: apple) to more accurately represent the target object (example: orange).<center>![](https://i.imgur.com/pg8rglY.png =400x)</center>

Additionally, this 4th channel can be saved as a separate grayscale image to visualize what features the generator considers to be useful. We test this hypothesis on the apple-orange dataset. This was accomplished using the pseudo code below:

```
criterionEnc=torch.nn.L1Loss()

embedding=torch.zeros(1,1,256,256)
embed_loss= (criterionEnc(embed_a,embedding) 
             + criterionEnc(embed_b,embedding)) * lambda_reg
             
loss_G = loss_G_A + loss_G_B 
       + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
       + embed_loss

```
embed_a and embed_b contains the results of the embedded channel after being passed through the respective generators.

## RESULTS
We ran the various modifications of the CycleGan and the baseline for 100 epochs each. We ran it for 100, instead of the default 200 epochs due to the resource and the time constraint, and the fact that we got reasonable results with 100 epochs for our dataset.
Since in CycleGan the generator tries to translate from one dataset to another, it is tough to use a metric to define how good the model performed. The images of the two datasets do not need to have the same background or even similiar images (there are images of orange juice or oranges growing, compared to apples in a shop etc). We thought of several methods such as metric learning (siamese networks) to identify the closeness the generated oranges have to the real orange image. But in the end decided that a qualitative analysis of the images would provide the most fruitful analysis of different models and the type of changes the CycleGan made.

To show the results, we picked a few images from the test set, to capture the most varience and then compared the results of those, to see what type of images a model has difficulty in translating.
The models chosen are:
1) Baseline cyclegan with cycle consistency weight of 10.
2) CycleGan with only cycle-consistency weight decay as explained above
3) CycleGan with only the feature extraction comparision
4) CycleGan with the combination of weight decay and feature extraction
5) CycleGan with embedded layer, with a cycle-consistency weight of 5.
 
 &nbsp; &nbsp; &nbsp; real &nbsp; &nbsp; &nbsp; &nbsp; baseline &nbsp; &nbsp; &nbsp;  F_extract &nbsp; &nbsp; &nbsp; w_decay &nbsp; combined &nbsp;embedded img+channel
![](https://i.imgur.com/bii801a.png =90x)![](https://i.imgur.com/P96atc7.png =90x)![](https://i.imgur.com/M2rtXfn.png =90x)![](https://i.imgur.com/TcER1Qf.png =90x)![](https://i.imgur.com/3elOOTO.png =90x)![](https://i.imgur.com/6tMxQij.png =90x)![](https://i.imgur.com/YGtfLNK.jpg =90x)
![](https://i.imgur.com/Zv72UfH.png =90x)![](https://i.imgur.com/ozjzDt3.png =90x)![](https://i.imgur.com/xs3QchM.png =90x)![](https://i.imgur.com/eiz6hpx.png =90x)![](https://i.imgur.com/WWkyHLY.png =90x)![](https://i.imgur.com/ZgRZORE.png =90x)![](https://i.imgur.com/xdrqpq6.jpg =90x)
![](https://i.imgur.com/jazStKs.png =90x)![](https://i.imgur.com/0Sizhv9.png =90x)![](https://i.imgur.com/DUWJGaZ.png =90x)![](https://i.imgur.com/EKGSd8G.png =90x)![](https://i.imgur.com/jIiVjkY.png =90x)![](https://i.imgur.com/zeL40uX.png =90x)![](https://i.imgur.com/0Rw4QWX.jpg =90x)
![](https://i.imgur.com/fjt3s6s.png =90x)![](https://i.imgur.com/cK5C1EQ.png =90x)![](https://i.imgur.com/CbK2AbB.png =90x)![](https://i.imgur.com/DxxqnKF.png =90x)![](https://i.imgur.com/xh0mYDi.png =90x)![](https://i.imgur.com/bN3eMaq.png =90x)![](https://i.imgur.com/hckpYvM.jpg =90x)

As we can see in the above results, the performance, especially in the colour mapping is very different in each of the models.
We can see that having only the cycle-consistancy loss decay does not yield good images, this might be because decaying the cycle consistency reduces the ability of the CycleGan to recreate the image, maybe due to the loss of certain background information during mapping. We can also notice that combinining the feature level cycle consistancy  with the cycle-consistency loss, though improving the results, does not perform as good as the baseline results, the recreation is still not as good (can be seen with the backfround of the images). The overall way/amount by which the weight is decayed could be changed to see if there is some modifications in the results. While training we divided the cycle-consistency weight by 2 after every 25th epoch.

When we look at the results of the model which tries to extract only the features, that is uses the feature level cycle consistancy, as compared to the combined loss, we find out that the colour mapping is much richer and close to an orange. We can see that other than the second set of results, the colour mapping done by using only the gamma decay loss function actually outperforms the baseline model.

Looking at the results of the embedded CycleGan, we can see the role of the embedded layer, and the type of features which it records. This layer did help the cyclegan to recreate the original image, even when the cycle-consitancy weight is reduced. Due to this we are able to allow the model to recreate as much of the orange as possible (richer color and better texture and shadow details), while mainting the background details, and better showing the details such as the water droplets in the last set of images. Though the colour mapping done by this model is the best out of the lot we tested, the model still did not translate the shape of the oranges, we can see that the oranges still have the shape of the apples. There are a couple of hypothesis for this:
1) Varying the cycle-consistency loss and the weights for the embbeded layers, we might be able to have a more of a tradeoff between recreating the original image and alowing for better shape translations.
2) The dataset of apples and oranges contain a lot of inconsistent images and only a 1000 training images overall. There are a lot of images of orange juice, people holding oranges, cut fruits, fruits in a shop/plates. Thus there is a lot of variety of images in an overall small dataset. This might be an issue while mapping the shape of an apple to an orange.

## Conclusion
This project was a successful proof of concept for testing some of the newer ideas for improving the performance of CycleGANs. Embedded CycleGANs seem to be the most promising improvement as it not only learns color mapping better, it also is able to capture the texture due to the embedded channel. In our project, we limited the embedded channel to one, but in theory, we can add as many additional channels as required. Moreover, by tuning the cycle consistency loss weights appropriately, we may be able to reduce the focus on exact recreation of the image as long as the recreated image has the right features. This would give the CycleGAN more freedom to change the shape of the input object to better match the target object. Taking a combination of embedded CycleGANs, Feature Level Cycle Consistency, Cycle Consistency Weight Decay with properly tuned hyper parameters may prove to be the next step in CycleGANs, with their combined ability to learn accurate color mapping, texture, and the ability to change the object's shape.

### References
[CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
[CycleGAN with Better Cycles](https://ssnl.github.io/better_cycles/report.pdf)
[Embedded CycleGAN](https://ieeexplore-ieee-org.tudelft.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=8803082)
























