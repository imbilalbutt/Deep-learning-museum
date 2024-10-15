class RNN:
    def __init__(self, input_size, hidden_size, output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False

    def forward(self, x_batch: ndarray) -> ndarray:
        assert_dim(ndarray, 3)
        x_out = x_batch
        for layer in self.layers:
    x_out = layer.forward(x_out)
    return x_out
    # def forward(self, input_tensor):
    #     trajectories = [input_tensor.detach().numpy()] #save the initial value of the trayectory
    #                                               #(we use detach to don't disturve in the computational graph)
    #
    #     for block in self.com_residual_blocks: #We apply block by block of our residual_blocks
    #         input_tensor = block(input_tensor)
    #
    #         trajectories.append(input_tensor.detach().numpy())   #We save the trayecty in that specific block
    #
    #     pred = self.linear_layer(input_tensor)
    #     pred_norm = torch.sigmoid(pred)
    #     return pred_norm , trajectories

