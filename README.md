# MPI_SAGAN -  MIDI Piano Image Generative Adversarial Network using Self-Attention 
____

This project is the generating music using self attention gan.

Out result is <a href="https://github.com/Malgus1995/MPI_SAGAN#self-attention-gan-result" >here</a>.

Before training the model, we convert midi file to grayscale midi image.


 


<img src='./piano_roll.png' height="106" width="270"><br>
piano roll midi file <br>

<img src='./midi_img/alb_esp6_Piano_1.png'><br>
gray scale midi image<br> 

### training the gan without thresh hold / training the gan applying 0.3 thresh / training the gan applying 0.5 thresh
![training_process ](./training_process_gif.gif) ![training_process 0_3](./training_process_gif_thresh_0_3.gif) ![training_process 0_5](./training_process_gif_thresh_0_5.gif) <br>

## Models arthitecture
___

![discriminator ](./discriminator.JPG) <br>

![gan ](./gan_model_arhitecture.JPG) <br>

![sagan ](./self_attention_gan_model_arhitecture.JPG) <br>

## Data source
___
<strong> GTZAN Dataset - Music Genre Classification</strong> <br>
Link : https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification <br>
<strong> Classical Music MIDI</strong> <br>
Link : https://www.kaggle.com/soumikrakshit/classical-music-midi <br>
<strong> Magenta Datasets </strong> <br>
Link : https://magenta.tensorflow.org/datasets/ <br>
<strong> Music MIDI Collection</strong> <br>
Link : https://www.kaggle.com/programgeek01/anime-music-midi

## GAN Result
___

### The DC gan model which trained 450 epoch generate fake music
![dcgan450](./vanilagan450.gif) <br>
youtube link: https://youtu.be/HZKP8TjhIk8

### The DC gan model which trained 1000 epoch generate fake music
![dcgan1000](./valinagan1000.gif) <br>
youtube link: https://youtu.be/apQXf4pcT2I

## Self attention GAN Result
___
### The self attention gan model which trained 150 epoch generate fake music
![sagan150](./sa150_0.gif) <br>
youtube link: https://youtu.be/QxlSpPadJnE

### The self attention gan model which trained 190 epoch generate fake music
![sagan190](./test190_0.gif) <br>
youtube link: https://youtu.be/5XvYgP_0rw0

### The self attention gan model which trained 420 epoch generate fake music
![sagan420](./test420_0.gif) <br>
youtube link: https://youtu.be/nfh7_8glCIM

### The self attention gan model which trained 430 epoch generate fake music
![sagan430](./test430_0.gif) <br>
youtube link: https://youtu.be/dNytCDL6Ss4
___
