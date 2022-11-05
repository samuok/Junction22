# Junction22

Instructions of usage:

Due to some problems with a library we used, we couldn't make the program run from a .exe file. So to run the program, start by cloning the project. Once that's done you need to create a database on mariaDB using the script found in the github repository. After this just run the file: main.py

Once the program has opened press Admin to open the data managment window. Get started by adding data to the simulation. This can be done by pressing the "Add Data" button and waiting for the text "Data Added" to appear in the statusbar at the bottom. The number of added data points can be changed by changing the number in the box beside the button. After this define the split(%) percentage to split the data into training, validation and testing. After finding a suitable split press the split button and wait for the statusbar to show split completed. After this press the training button to train the model. Wait for the text Training complete before going to the next step. After this press the prediction button to get the results of the simulation. If you want to save the model press save model to save the currently learned algorithm to memory. To later retrieve this model press load model what will retrieve a previously saved model from memory.

If you want to change the training models parameters you can press the "best parameter" button to check for the best parameters for the current simulation. Be warned this will do 500 iterations of the simulation and will take a considerable time with any large data set larger than a couple of thousand data points. After the iterations are complete the best found parameters will be seen in the best parameters box and can be set as the training variables by changing the values next to the training button.