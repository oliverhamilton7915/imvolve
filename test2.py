from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def hill_climb():
    # 1. load up goal image
    goal_image = Image.open('brand.jpg')  # a small picture of the firefox logo
    goal_array = np.array(goal_image) / 255.0

    # 2. load up starting candidate
    candidate_array = np.ones(shape=goal_array.shape)

    plt.imshow(goal_array)
    plt.show()
    plt.imshow(candidate_array)
    plt.show()


hill_climb()