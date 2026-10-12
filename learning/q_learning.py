import json
import os
import random


class QLearningAgent:
    def __init__(self):
        # 0 = accelerate
        # 1 = turn left
        # 2 = turn right
        # 3 = brake
        self.actions = [0, 1, 2, 3]

        self.learning_rate = 0.10
        self.discount_factor = 0.95

        # Start with some exploration, but don't make the AI random too often.
        self.epsilon = 0.08
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.02

        self.q_table = {}

        project_folder = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.memory_file = os.path.join(
            project_folder,
            "ai_memory.json"
        )

        self.load_q_table()

    def get_state(self, sensors, speed, heading_error=0, lateral_offset=0):
        """
        Convert the car's situation into a small discrete state.

        sensors:
            distances from the car to the track boundary.

        speed:
            current car speed.

        heading_error:
            difference between where the car is pointing
            and where the track wants it to point.
        """

        sensor_levels = []

        for distance in sensors:

            if distance < 25:
                level = 0

            elif distance < 60:
                level = 1

            elif distance < 100:
                level = 2

            else:
                level = 3

            sensor_levels.append(level)

        # Heading
        if heading_error < -30:
            heading_level = 0

        elif heading_error < -10:
            heading_level = 1

        elif heading_error <= 10:
            heading_level = 2

        elif heading_error <= 30:
            heading_level = 3

        else:
            heading_level = 4

        # Lateral position relative to the track centerline.
        if lateral_offset < -24:
            lateral_level = 0
        elif lateral_offset < -8:
            lateral_level = 1
        elif lateral_offset <= 8:
            lateral_level = 2
        elif lateral_offset <= 24:
            lateral_level = 3
        else:
            lateral_level = 4

        # Speed
        if speed < 0.5:
            speed_level = 0

        elif speed < 2:
            speed_level = 1

        elif speed < 3.5:
            speed_level = 2

        elif speed < 4.5:
            speed_level = 3

        else:
            speed_level = 4

        state_values = sensor_levels + [
            heading_level,
            speed_level,
            lateral_level
        ]

        return ",".join(
            map(str, state_values)
        )

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = [
                0.0 for _ in self.actions
            ]

        return self.q_table[state]

    def choose_action(self, state):

        values = self.get_q_values(state)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation
        highest_value = max(values)

        best_actions = [
            action
            for action, value in enumerate(values)
            if value == highest_value
        ]

        return random.choice(best_actions)

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done=False
    ):

        current_values = self.get_q_values(state)

        old_value = current_values[action]

        if done:
            future_value = 0.0

        else:
            future_value = max(
                self.get_q_values(next_state)
            )

        target = (
            reward
            + self.discount_factor * future_value
        )

        current_values[action] = (
            old_value
            + self.learning_rate
            * (target - old_value)
        )

    def reduce_exploration(self):

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )

    def save_q_table(self):

        try:

            with open(
                self.memory_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    {
                        "q_table": self.q_table,
                        "epsilon": self.epsilon
                    },
                    file,
                    indent=2
                )

            print(
                "AI memory saved!",
                f"Learned states: {len(self.q_table)}"
            )

        except OSError as error:

            print(
                f"Could not save AI memory: {error}"
            )

    def load_q_table(self):

        if not os.path.exists(
            self.memory_file
        ):

            print(
                "Starting with a new Q-table."
            )

            return

        try:

            with open(
                self.memory_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            if not isinstance(data, dict):

                print("Invalid AI memory.")

                return

            saved_table = data.get(
                "q_table",
                {}
            )

            if not isinstance(
                saved_table,
                dict
            ):

                print("Invalid Q-table.")

                return

            self.q_table = {}

            for state, values in saved_table.items():

                if not isinstance(
                    state,
                    str
                ):
                    continue

                if not isinstance(
                    values,
                    list
                ):
                    continue

                if len(values) != len(
                    self.actions
                ):
                    continue

                if not all(
                    isinstance(
                        value,
                        (int, float)
                    )
                    for value in values
                ):
                    continue

                self.q_table[state] = [
                    float(value)
                    for value in values
                ]

            saved_epsilon = data.get(
                "epsilon",
                self.epsilon
            )

            if isinstance(
                saved_epsilon,
                (int, float)
            ):

                self.epsilon = max(
                    self.min_epsilon,
                    min(
                        1.0,
                        float(saved_epsilon)
                    )
                )

            print(
                "AI memory loaded!",
                f"Learned states: {len(self.q_table)}"
            )

        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError
        ) as error:

            self.q_table = {}

            print(
                f"Could not load AI memory: {error}"
            )