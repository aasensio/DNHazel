test.py
e.g.
run test.py --input=keepsize_relu_x2_wrap --depth=15 --kernels=64 --model=keepsize --activation=relu --action=movie
--------

    --input=keepsize_relu 
        Define the input file with the network. This will be the same that was used as --output when training. A `../training/networks` will be prefixed, which is the usual place for trained networks.
    --depth=15
        Number of residual blocks used in the network. This number will affect differently depending on the topology of the network
    --model={keepsize,encdec}
        `keepsize` is a network that maintains the size of the input and output, with an eventual upsampling at the end in case of superresolution
        `encdec` is an encoder-decoder network
    --padding={zero,reflect}
        `zero` uses zero padding for keeping the size of the images through the network. This might produce some border artifacts 
        `reflect` uses reflection padding, which strongly reduces these artifacts
    --activation={relu,elu}
        Type of activation function to be used in the network, except for the last convolutional layer, which uses a linear activation
    --action={cube,movie,large_frame}
        `cube`: display 100 images and allows the user to interactively analyze the result of the network
        `movie`: generates a couple of movies for several frames
        `large_frame`: not yet implemented
