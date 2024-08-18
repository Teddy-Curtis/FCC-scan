import torch
import numpy as np


class ParametricModel(torch.nn.Module):
    """This has the generic functions that all the pNN's will use"""

    def __init__(self):
        super(ParametricModel, self).__init__()

    def resampleBkgMasses(self):
        # Finds what masses are available and random selects for each bkg
        return 0


###############################################################################
##########################       Affine Model       ###########################
###############################################################################
# Define model
class AffineLayer(torch.nn.Module):
    def __init__(self, input_shape, num_masses):
        super(AffineLayer, self).__init__()
        # torch.manual_seed(12345)

        self.linear1 = torch.nn.Linear(num_masses, input_shape)

        self.linear2 = torch.nn.Linear(num_masses, input_shape)

    def forward(self, x, masses):
        mass1_output = self.linear1(masses)
        mass2_output = self.linear2(masses)
        x = x * mass1_output

        x = x + mass2_output

        return x


# Define model
class DenseAffineLayer(torch.nn.Module):
    def __init__(self, prev_shape, output_shape, drop_rate, num_masses):
        super(DenseAffineLayer, self).__init__()
        # torch.manual_seed(12345)
        self.dropout = torch.nn.Dropout(p=drop_rate)
        self.linear = torch.nn.Linear(prev_shape, output_shape)
        self.activation = torch.nn.ELU()
        self.affine = AffineLayer(output_shape, num_masses)

    def forward(self, x, masses):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.affine(x, masses)
        return x


# Define model
class AffineModel(torch.nn.Module):
    def __init__(self, settings):
        super(AffineModel, self).__init__()
        # torch.manual_seed(12345)

        previous_output_shape = settings["input_features"]

        # Define the fully connected layers
        self.fc_process = torch.nn.ModuleList()
        for layer_param in settings["fc_params"]:
            drop_rate, H = layer_param
            layer = DenseAffineLayer(
                previous_output_shape, H, drop_rate, settings["num_masses"]
            )
            self.fc_process.append(layer)
            previous_output_shape = H

        # Final output layer
        self.output_mlp_linear = torch.nn.Linear(
            previous_output_shape, settings["output_classes"]
        )

    def forward(self, x, masses):

        for layer in self.fc_process:
            x = layer(x, masses)

        x = self.output_mlp_linear(x)

        return x


###############################################################################
##########################       FeedForward Elu      #########################
###############################################################################
# Define model
class SimpleMLPElu(torch.nn.Module):
    def __init__(self, settings):
        super(SimpleMLPElu, self).__init__()
        # torch.manual_seed(12345)
        self.model_name = "SimpleMLPElu"

        previous_output_shape = settings["input_features"]

        # Define the fully connected layers
        self.fc_process = torch.nn.ModuleList()
        for layer_param in settings["fc_params"]:
            drop_rate, H = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Dropout(p=drop_rate),
                torch.nn.Linear(previous_output_shape, H),
                torch.nn.ELU(),
            )
            self.fc_process.append(seq)
            previous_output_shape = H

        # Final output layer
        self.output_mlp_linear = torch.nn.Linear(
            previous_output_shape, settings["output_classes"]
        )

    def forward(self, x, masses):
        x = torch.cat([x, masses], axis=-1)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)

        return x


###############################################################################
###################       FeedForwark Leaky Relu BN       #####################
###############################################################################
# Define model
class MLPLeakyReluBatchNorm(torch.nn.Module):
    def __init__(self, settings):
        super(MLPLeakyReluBatchNorm, self).__init__()
        # torch.manual_seed(12345)
        self.model_name = "MLPLeakyReluBatchNorm"

        previous_output_shape = settings["input_features"]

        # Define the fully connected layers
        self.fc_process = torch.nn.ModuleList()
        for layer_param in settings["fc_params"]:
            drop_rate, H = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Dropout(p=drop_rate),
                torch.nn.Linear(previous_output_shape, H),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(H),
            )
            self.fc_process.append(seq)
            previous_output_shape = H

        # Final output layer
        self.output_mlp_linear = torch.nn.Linear(
            previous_output_shape, settings["output_classes"]
        )

    def forward(self, x, masses):
        x = torch.cat([x, masses], axis=-1)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)

        return x


###############################################################################
###################       FeedForward Relu        #####################
###############################################################################
# Define model
class MLPRelu(torch.nn.Module):
    def __init__(self, settings):
        super(MLPRelu, self).__init__()
        # torch.manual_seed(12345)

        previous_output_shape = settings["input_features"]

        # Define the fully connected layers
        self.fc_process = torch.nn.ModuleList()
        for layer_param in settings["fc_params"]:
            drop_rate, H = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Dropout(p=drop_rate),
                torch.nn.Linear(previous_output_shape, H),
                torch.nn.LeakyReLU(),
            )
            self.fc_process.append(seq)
            previous_output_shape = H

        # Final output layer
        self.output_mlp_linear = torch.nn.Linear(
            previous_output_shape, settings["output_classes"]
        )

    def forward(self, x, masses):
        x = torch.cat([x, masses], axis=-1)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)

        return x


###############################################################################
###################       Non-pNN FeedForward Relu        #####################
###############################################################################
# Define model
class NonpNNMLPRelu(torch.nn.Module):
    def __init__(self, settings):
        super(NonpNNMLPRelu, self).__init__()
        # torch.manual_seed(12345)

        previous_output_shape = settings["input_features"]

        # Define the fully connected layers
        self.fc_process = torch.nn.ModuleList()
        for layer_param in settings["fc_params"]:
            drop_rate, H = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Dropout(p=drop_rate),
                torch.nn.Linear(previous_output_shape, H),
                torch.nn.LeakyReLU(),
            )
            self.fc_process.append(seq)
            previous_output_shape = H

        # Final output layer
        self.output_mlp_linear = torch.nn.Linear(
            previous_output_shape, settings["output_classes"]
        )

    def forward(self, x, masses):
        # x = torch.cat([x, masses], axis=-1)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)

        return x
