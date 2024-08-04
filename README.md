# Autonomouse
This application fully automates the role of a mouse by replacing it exclusively with head movements and blinks.

## Guide to Usage
For maximum functionality, ensure that your surroundings are well-lit and that your device camera is as close as possible to being level with your eyes.

Download the files and simply open the run.bat file. The application should get started immediately. 

You will see a small window where you can view the live camera recording of yourself, as well as the x, y, and z coordinates of your head. Tilt your head in the direction corresponding to the direction that you intend to move your mouse. If you tilt your head enough, the speed of the mouse in that particular direction will increase.

![image](https://github.com/user-attachments/assets/dc65213d-d1c1-4c9f-8d3b-553fa99a6338)

To emulate a left-click of the mouse, close your eyes for a brief moment (equivalent to 4 frames worth of time), approximately 1.5 - 2 seconds. To emulate a right-click of the mouse, close your eyes for twice that amount of time. The extended duration of the "blinks" is meant to signal their intentionality as commands to the program.

### Improvements Needed
* mouse (in)sensitivity can be irritating and requires further tuning. Mouse may sometimes be too slow when moving across a page but too quick when trying to move by a few pixels.
* current command durations (blinks) are too long - far slower than using a handheld mouse. Working on incorporating winks for right-click. However, these are significantly harder to register.   
