# characterrnn
Character-level language modelling using a Recurrent Neural Network.

TODOs:
1. accumulate the layer deltas rather than the weight updates, this should be
   faster.
2. dropout regularization.
3. beam search for text generation/sampling.
4. save/load the RNN to disk.
5. check the L2 magnitudes of the weight updates, clip the weights if necessary.

