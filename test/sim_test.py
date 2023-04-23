import numpy as np
import pygame
import sys
import os

root_path: str = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir))
print(root_path)
sys.path.append(root_path)
from simulator import simulator  # noqa

sim_wall = simulator.Walls()
sim_wall.append([-200, -50], [150, -40])
sim_wall.append([0, 130], [170, 0])
sim_wall.append([20, 170], [-150, -40])

wall_arr = sim_wall.get()

init_pos = np.array([0, 0, 0])
num_lazer = 16
sim = simulator.Simulator(wall_arr, init_pos, num_lazer)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    sim.move(0, 0, 0.01)
    # sim.measure()
    sim.draw()
