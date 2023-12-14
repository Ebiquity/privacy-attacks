### Perform a Membership Inference Attack on Synthesized data

Run 'parse_data.py' to convert a csv file to an npz and json file with the format the mia code expects <br><br>
Run python train_victim_model.py --data Network
Run 'dist_info.py' <br><br>
Code for the Membership Inference Attack can be found here: <br>
https://github.com/JayoungKim408/MIA/tree/master <br><br>
Perform the attacks with the following commands, replace Network with the name of your dataset <br>
python main.py --data Network --attack fbb <br>
python main.py --data Network --attack wb <br><br>
Code tested on python 3.9.2<br>
See requirements.txt for dependencies
