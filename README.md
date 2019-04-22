# Optical_Player_Tracking_System_For_Soccer
A new and non-invasive method of tracking Players in a Soccer Match.

Currently This system starts with Manual box-bounding the players in an offline video, but the whole system will be redefined soon to bring in complete automation ,json format for storing location coordinates and support for live streaming videos / camera input will be added.


1. Clone this github repository or download the python file.
2. run `$ python opts_final.py [video_path]`
3. Manually draw boxes for players one by one and press 'a'. Press 'q' after bounding the final player to be tracked and Hit 'Enter/Return'.
4. Three windows pop with different results
5. A text file will be saved in the current directory with x and y locations of players according to program's Window size.
