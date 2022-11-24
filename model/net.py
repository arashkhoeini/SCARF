import torch

class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, layers_dim, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        layers_dim = layers_dim.copy()
        layers_dim.insert(0, input_dim)
        layers_dim.append(output_dim)
        for i in range(len(layers_dim)-1):
            layers.append(torch.nn.Linear(layers_dim[i], layers_dim[i+1]))
            if i+1 != len(layers_dim):
                layers.append(torch.nn.ReLU(inplace=True))
                layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)

class Net(torch.nn.Module):

    def __init__(self, input_dim, n_classes, configs):
        super().__init__()
        self.encoder = MLP(input_dim, configs.encoder_dims, configs.rep_size)
        self.pretraining_head = MLP(configs.rep_size, configs.pretraining_head_dims, configs.rep_size)
        self.prediction_head = MLP(configs.rep_size, configs.prediction_head_dims, n_classes)

        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, X):
        X = X.to(torch.float32)
        features = self.encoder(X)
        z = self.pretraining_head(features)
        pred_raw = self.prediction_head(features)
        pred = self.softmax(pred_raw)
        return features, z, pred_raw, pred