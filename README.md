# RNN-Hand-Written-Digit-Recognition
Hand written digit recognition using LSTM( long short term memory)

Instead of LSTM, you can use CuDNNLSTM which is available in gpu version of tensorflow, do use it if you have an access to gpu.
CuDNNLSTM greatly reduces training time with no signifacnt or none at all decrease in accuracy.
CuDNNLSTM already has an in built activation function but it can be overridden by writing code the same way as we do for regular lstm.
