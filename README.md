# MPI_SAGAN
____
This project is the generating music using self attention gan.

Before training the model, we convert midi file to graysacle midi image.

<img src='./piano_roll.png'><br>
piano roll midi file <br>

<img src='./midi_img/alb_esp6_Piano_1.png'><br>
gray scale midi image<br>

##Data source
___
<strong> GTZAN Dataset - Music Genre Classification</strong> <br>
Link : https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification <br>
<strong> Classical Music MIDI</strong> <br>
Link : https://www.kaggle.com/soumikrakshit/classical-music-midi <br>
<strong> Classical Music MIDI</strong> <br>
Link : https://www.kaggle.com/soumikrakshit/classical-music-midi <br>



## GAN Result
___


## Self attention GAN Result
___
###The model which trained 150 epoch generate fake music
<audio controls>
  <source src="./best_result/sa150_0.3test_good.mp3" type="audio/mpeg">

</audio><br>

###The model which trained 150 epoch generate fake music
<audio controls>
  <source src="horse.mp3" type="audio/mpeg">

</audio>