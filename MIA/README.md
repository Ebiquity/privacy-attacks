### Perform a Membership Inference Attack on Synthesized data
Run 'parse_data.py' to convert a csv file to an npz and json file with the format the mia code expects <br><br>
Run 'dist_info.py' <br><br>
Code for the Membership Inference Attack can be found here: <br>
https://github.com/JayoungKim408/MIA/tree/master <br><br>
Perform the attacks with the following commands, replace Network with the name of your dataset <br>
main.py --data Network --attack fbb <br>
main.py --data Network --attack wb <br>
