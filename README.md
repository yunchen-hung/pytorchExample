# pytorchExample

Instructions: To run the model, enter `conda install -c pytorch pytorch` then `python main.py --model DAN` in the terminal. This should print out the test and dev accuracy, and the test and dev loss. The model for 2b. is printed as "2 layers, own embedding", while the original pretrained embedding layer is simply named "2 layers"

The goal of this project was to implement a Deep Averaging Network (DAN) model to compare performance against a Bag Of Words (BOW) model. To complete this task I used PyTorch to created a basic DAN model with pretrained embedding layers. Some specific things I did was created a collate function to handle text inputs of various lengths and padded with 0 values. This model had an train accuracy of 0.937, and a dev accuracy of 0.950, which was odd but ultimately I was unable to locate the problem. However, I found that the loss was decreasing over time.
