"""Entry point used by the ARC autograder."""

from arc_strategies import make_agent_predictions


class ArcAgent:
    """Collect the training examples and ask the strategy code for predictions."""

    def __init__(self):
        pass

    def make_predictions(self, arc_problem):
        training_pairs = []

        for arc_set in arc_problem.training_set():
            output_data = arc_set.get_output_data()
            if output_data is not None:
                training_pairs.append(
                    (
                        arc_set.get_input_data().data(),
                        output_data.data(),
                    )
                )

        test_input = arc_problem.test_set().get_input_data().data()
        return make_agent_predictions(training_pairs, test_input)
