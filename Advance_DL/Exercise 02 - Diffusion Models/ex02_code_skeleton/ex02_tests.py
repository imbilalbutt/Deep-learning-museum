import unittest
import torch
import os
from numpy.testing import assert_almost_equal

from ex02_diffusion import Diffusion, linear_beta_schedule

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TestDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.test_values = torch.load("ex02_test_values.pt")
        self.scheduler = lambda x: linear_beta_schedule(0.001, 0.02, x)
        self.img_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_q_sample(self):
        local_values = self.test_values["q_sample"]
        diffusor = Diffusion(timesteps=local_values["timesteps"],
                             get_noise_schedule=self.scheduler, img_size=self.img_size)

        output = diffusor.q_sample(x_zero=local_values["x_zero"].to(self.device),
                                   t=local_values["t"].to(self.device), noise=local_values["noise"].to(self.device))
        assert_almost_equal(local_values["expected_output"].numpy(), output.to(torch.device("cpu")).numpy(), decimal=5)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
