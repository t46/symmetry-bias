"""
Class that unifies the functions for the experiment
"""
from torch.nn import functional as F
from torch import nn
from torch import optim

import numpy as np
import torch
import copy

import sys

sys.path.append('..')
from models.simple_models import ForwardModel, BackwardModel
from utils.metrics import calc_accuracy


class ExperimentManager():
    """
    Class to run trials for experiments.
    """
    def __init__(self, seed, data_dim) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.X = torch.normal(0, 1, size=(2, data_dim))
        self.Y = torch.arange(2)
        # one-hot encoding is used for the input to backward model.
        self.Y_one_hot = F.one_hot(self.Y,
                                   num_classes=2).type(torch.FloatTensor)

        self.forward_model = ForwardModel(in_dim=data_dim, out_dim=2)
        # self.backward_model is trained in training loop.
        # This is the model for considering symmetry bias.
        # self.backward_model_test_only is not trained during test.
        # This is the model for comparision.
        self.backward_model = BackwardModel(in_dim=2, out_dim=data_dim)
        self.backward_model_test_only = copy.deepcopy(self.backward_model)

        self.cos_sim_calculator = nn.CosineSimilarity(dim=1, eps=1e-8)

        self.loss_function_forward = nn.CrossEntropyLoss()
        self.loss_function_backward = nn.MSELoss()

        self.optimizer_forward = optim.SGD(self.forward_model.parameters(),
                                           lr=0.1)
        self.optimizer_backward = optim.SGD(self.backward_model.parameters(),
                                            lr=0.1)

        self.convergence_threshold = 1e-3

    def run_train(self, train_iterations=1000) -> None:
        """Method to run training.

        Args:
            train_iterations (int, optional):
            The number of training iterations. Defaults to 1000.
        """
        for _ in range(train_iterations):
            idx = np.random.choice(2)
            x = self.X[idx].unsqueeze(0)
            y = self.Y[idx].unsqueeze(0)

            # Forward model
            output_forward = self.forward_model(x)
            prob_y_given_x = F.softmax(output_forward, dim=1).detach()
            loss_forward = self.loss_function_forward(output_forward, y)
            self.optimizer_forward.zero_grad()
            loss_forward.backward()
            self.optimizer_forward.step()
            # probs_y0_given_x.append(F.softmax(output_forward,
            #                         dim=1)[0][0].detach().numpy())
            # probs_y1_given_x.append(F.softmax(output_forward,
            #                         dim=1)[0][1].detach().numpy())

            # Backward model
            y_one_hot = self.Y_one_hot[idx].unsqueeze(0)
            output_backward = self.backward_model(y_one_hot)
            # if cosine similarity between output_backward and
            # the input x is high, the probability p(X|Y) is
            # considered to be high.
            cos_sim = self.cos_sim_calculator(output_backward,
                                              self.X).unsqueeze(0)
            prob_x_given_y = F.softmax(cos_sim, dim=1)
            loss_backward = self.loss_function_backward(prob_x_given_y,
                                                        prob_y_given_x)
            self.optimizer_backward.zero_grad()
            loss_backward.backward()
            self.optimizer_backward.step()
            # probs_x0_given_y.append(F.softmax(cos_sim,
            #                         dim=1)[0][0].detach().numpy())
            # probs_x1_given_y.append(F.softmax(cos_sim,
            #                         dim=1)[0][1].detach().numpy())

    def run_test(self, test_iterations=500):
        """Method to run test.

        Args:
            test_iterations (int, optional):
            The number of test iterations. Defaults to 500.
        """
        self.test_losses_backward = []
        self.accuracies_backward = []
        self.probs_x0_given_y = []
        self.probs_x1_given_y = []
        self.steps_to_converge = test_iterations

        self.test_losses_backward_test_only = []
        self.accuracies_backward_test_only = []
        self.probs_x0_given_y_test_only = []
        self.probs_x1_given_y_test_only = []
        self.steps_to_converge_test_only = test_iterations

        optimizer_backward_test_only = \
            optim.SGD(self.backward_model_test_only.parameters(), lr=0.1)

        for step in range(test_iterations):
            idx = np.random.choice(2)
            x = self.X[idx].unsqueeze(0)
            y_one_hot = self.Y_one_hot[idx].unsqueeze(0)
            y = self.Y[idx]

            # Backward model with symmetry bias
            output_backward = self.backward_model(y_one_hot)
            cos_sim = self.cos_sim_calculator(output_backward,
                                              self.X).unsqueeze(0)
            loss_backward = self.loss_function_backward(output_backward, x)
            self.optimizer_backward.zero_grad()
            loss_backward.backward()
            self.optimizer_backward.step()
            test_loss = loss_backward.detach().numpy()
            self.test_losses_backward.append(test_loss)
            prob_x0_given_y = F.softmax(cos_sim, dim=1)[0][0].detach().numpy()
            prob_x1_given_y = F.softmax(cos_sim, dim=1)[0][1].detach().numpy()
            self.probs_x0_given_y.append(prob_x0_given_y)
            self.probs_x1_given_y.append(prob_x1_given_y)
            pred = np.argmax([prob_x0_given_y, prob_x1_given_y])
            accuracy = calc_accuracy([pred], [y])
            self.accuracies_backward.append(accuracy)
            if self.steps_to_converge == test_iterations \
                and test_loss < self.convergence_threshold:
                self.steps_to_converge = step + 1
            # Backward model without symmetry bias
            output_backward_test_only = \
                self.backward_model_test_only(y_one_hot)
            cos_sim_test_only = \
                self.cos_sim_calculator(output_backward_test_only,
                                        self.X).unsqueeze(0)
            loss_backward_test_only = \
                self.loss_function_backward(output_backward_test_only, x)
            optimizer_backward_test_only.zero_grad()
            loss_backward_test_only.backward()
            optimizer_backward_test_only.step()
            test_loss_test_only = loss_backward_test_only.detach().numpy()
            self.test_losses_backward_test_only.append(
                test_loss_test_only
                )
            prob_x0_given_y_test_only = \
                F.softmax(cos_sim_test_only, dim=1)[0][0].detach().numpy()
            prob_x1_given_y_test_only = \
                F.softmax(cos_sim_test_only, dim=1)[0][1].detach().numpy()
            self.probs_x0_given_y_test_only.append(
                prob_x0_given_y_test_only
                )
            self.probs_x1_given_y_test_only.append(
                prob_x1_given_y_test_only
                )
            pred_test_only = np.argmax(
                [prob_x0_given_y_test_only, prob_x1_given_y_test_only])
            accuracy_test_only = calc_accuracy([pred_test_only], [y])
            self.accuracies_backward_test_only.append(accuracy_test_only)
            if self.steps_to_converge_test_only == test_iterations \
                and test_loss_test_only < self.convergence_threshold:
                self.steps_to_converge_test_only = step + 1
