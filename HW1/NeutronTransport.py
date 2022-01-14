import numpy as np
import pandas as pd


class Neutron(object):
    """
    neutron object
    """

    def __init__(self, initial_position: np.array, initial_velocity: np.array, p: float):
        """
        :param p: probability of escape
        :param initial_position: (1,2) array of angle (radians) and distance
        :param initial_velocity: (1,2) array of angle (radians) and magnitude
        """
        assert 1 > p > 0
        self.position = initial_position
        self.velocity = initial_velocity
        self.escaped = False
        self.time_in_circle = 0
        self.escape_prob = p

    def is_escaped(self):
        prob = np.random.uniform(0, 1, 1)
        if prob <= self.escape_prob:
            self.escaped = True

    def reflect(self, surface_orientation: float):
        """
        updates velocity based on reflection information
        :param surface_orientation: angle of orientation
        :return: new velocity (1,2) numpy array of angle and magnitude
        """
        ### need to find angle of incidence
        velocity_vec_xy = np.array([np.cos(self.velocity[0]) * self.velocity[1],
                                    np.sin(self.velocity[0] * self.velocity[1])])

        surface_vec_xy = np.array([np.cos(surface_orientation), np.sin(surface_orientation)])
        surface_vec_xy /= np.linalg.norm(surface_vec_xy)

        reflected_velocity_xy = -2 * (surface_vec_xy.dot(velocity_vec_xy) * surface_vec_xy - velocity_vec_xy)
        self.velocity = np.array([np.arctan2(reflected_velocity_xy[1], reflected_velocity_xy[0]),
                                  self.velocity[1]])

    def compute_time_to_distance(self, destination_distance: float) -> float:
        """

        :param destination_distance: when will our particle hit this distance
        :return: time it will take for particle to hit this point
        """

        initial_distance = self.position[1]
        velocity_magnitude = self.velocity[1]
        initial_orientation = self.position[0]
        velocity_orientation = self.velocity[0]
        a = -initial_distance * np.cos(initial_orientation + velocity_orientation) / velocity_magnitude
        b = np.sqrt((initial_distance * np.cos(initial_orientation + velocity_orientation)) ** 2 +
                    (destination_distance ** 2 - initial_distance ** 2)) / velocity_magnitude
        return np.squeeze(a + b)  ## time will always be positive

    def compute_orientation_at_distance(self, destination_distance: float) -> float:
        """

        :param destination_distance: where our particle will hit this distance
        :return: radians value
        """

        initial_distance = self.position[1]
        velocity_magnitude = self.velocity[1]
        initial_orientation = self.position[0]
        velocity_orientation = self.velocity[0]
        t = self.compute_time_to_distance(destination_distance)
        return np.squeeze(np.arctan2(
            initial_distance * np.sin(initial_orientation) + velocity_magnitude * t * np.sin(velocity_orientation),
            initial_distance * np.cos(initial_orientation) + velocity_magnitude * t * np.cos(velocity_orientation)))

    def simulate_refraction(self, radius: float) -> float:
        """
        simulate the possible refraction time of a particle inside a circle
        :param radius: with radius
        :return: time in seconds particle spent inside circle
        """

        while not self.escaped:
            t = self.compute_time_to_distance(radius)
            self.time_in_circle += t
            #### compute the value of the angle of the surface we're reflecting off
            ## 1. first create the radial vector to the new position
            new_position = self.compute_orientation_at_distance(radius)
            new_position_xy = np.array([radius * np.cos(new_position), radius * np.sin(new_position)])
            ### 2. compute the angle and update the resulting velocity
            tangent_vector_xy = np.array([1, -new_position_xy[0] / new_position_xy[1]])

            self.reflect(np.arctan2(tangent_vector_xy[1], tangent_vector_xy[0]))

            ### 3. update the position
            self.position = np.array([new_position, radius])

            ### 4. did we escape at this position
            self.is_escaped()

        return self.time_in_circle


if __name__ == "__main__":
    result_list = []
    radii = np.arange(1, 10)
    escape_probs = np.linspace(.1, 1, 10,endpoint=False)
    velocities = np.arange(1, 10)
    num_samples = 100000
    for radius in radii:
        for velocity in velocities:
            for escape_prob in escape_probs:
                times = []
                sampled_origins = np.random.uniform(low=[0, 0], high=[np.pi * 2, radius], size=(num_samples, 2))
                sampled_velocity_orientations = np.random.uniform(low=0, high=np.pi * 2, size=(num_samples,))
                for trial in range(num_samples):
                    neutron_particle = Neutron(sampled_origins[trial, :],
                                               np.array([sampled_velocity_orientations[trial], velocity]),
                                               p=escape_prob)
                    times.append(neutron_particle.simulate_refraction(radius))
                mean_time = np.mean(times)
                result_list.append(
                    {"radius": radius, "velocity": velocity, "escape_prob": escape_prob, "time": mean_time})

    result_dict = pd.DataFrame(result_list)
    print(result_dict)
