This directory should contain serialized Keras dialogue models, in tuples according to the following rough scheme:
     - an encoder-decoder model named 'test' will be spread across the following HDF5 files:
         (1) test\_train
         (2) test\_inference\_encoder
         (3) test\_inference\_decoder