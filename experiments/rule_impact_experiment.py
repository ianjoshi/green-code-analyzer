import time
import random
import os
from energibridge_executor import EnergibridgeExecutor
from pyEnergiBridge.api import EnergiBridgeRunner




class RuleImpactExperiment:
    """
    A class for running controlled energy consumption measurement experiments for the rules,
    in order to estimate their impact on the energy consumption of the system.
    The experiment includes a warm-up phase, task execution, and rest periods between runs.
    """

    def __init__(self, rules, num_runs=5, warmup_duration=300, rest_duration=60, measurement_duration=500):
        """
        Initializes the experiment with the necessary parameters.

        Parameters:
        - rules (dict): Dictionary of rules to be tested with rule name as key and a list of functions that execute an example of the rule as value.
        - num_runs (int): Number of times each task should be executed.
        - warmup_duration (int): Warm-up period (in seconds) before measurements.
        - rest_duration (int): Rest period (in seconds) between runs.
        - measurement_duration (int): Maximum duration of each measurement in seconds.
        """
        self.rules = rules
        self.num_runs = num_runs
        self.warmup_duration = warmup_duration
        self.rest_duration = rest_duration

        self.energibridge = EnergibridgeExecutor(max_measurement_duration=measurement_duration)
        self.energibridge_runner = EnergiBridgeRunner()
        
        # Create results directory if it doesn't exist
        os.makedirs("results/rule_impact_estimates", exist_ok=True)

    def run_experiment(self):
        """
        Orchestrates and runs the experiment sequence:
        1. Warns the user and prepares the environment.
        2. Warms up the CPU by running Fibonacci calculations.
        3. Runs each task multiple times in a shuffled order with rest intervals.
        """
        self._warn_and_prepare()
        self._warmup_fibonacci()
        self.energibridge.start_service()

        rule_names = list(self.rules.keys())
        random.shuffle(rule_names)
        for i in range(self.num_runs):
            for rule_name in rule_names:
                print(f"----- Run {rule_name} -----")
                output_file = f"results/rule_impact_estimates/{rule_name}.csv"
                rule_functions = self.rules[rule_name]

                print(f"Estimating the impact of the rule {rule_name} on the energy consumption of a system.")
                self._estimate_rule_impact(rule_functions, output_file)

                # Rest between runs except for the last iteration of last run
                if i < self.num_runs - 1:
                    print(f"Resting for {self.rest_duration} seconds before the next run...")
                    time.sleep(self.rest_duration)

        self.energibridge.stop_service()
        print("Experiment complete.")

    def _warn_and_prepare(self):
        """Provides instructions to the user to optimize system conditions before running the experiment."""
        print("WARNING: Before proceeding, please:")
        print("- Close all unnecessary applications.")
        print("- Kill unnecessary services.")
        print("- Turn off notifications.")
        print("- Disconnect any unnecessary hardware.")
        print("- Disconnect Wi-Fi.")
        print("- Switch off auto-brightness on your display.")
        print("- Set room temperature (if possible) to 25°C. Else stabilize room temperature if possible.")
        print("Press Enter to continue once the environment is ready.")
        input()

    def _warmup_fibonacci(self):
        """Runs Fibonacci calculations continuously for a specified duration to warm up the CPU."""
        print(f"Starting Fibonacci warm-up for {self.warmup_duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < self.warmup_duration:
            # Compute Fibonacci of 30 repeatedly
            self._fib(30) 

        print("Warm-up complete.")

    def _fib(self, n):
        """Computes the nth Fibonacci number iteratively."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _estimate_rule_impact(self, rule_examples, output_file):
        """Estimates the impact of the rule on the energy consumption of the system."""
        # Shuffle order of rule examples
        random.shuffle(rule_examples)
        
        # Start measuring the energy consumption
        energibridge_runner = EnergiBridgeRunner()
        energibridge_runner.start(output_file)
        
        # Execute each function containing an example of the rule
        for rule_example in rule_examples:
            rule_example()
        
        # Stop measuring the energy consumption
        energibridge_runner.stop()
        
          
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.long_loop import inefficient_loop
    rules = {"long_rule": [inefficient_loop]}
    experiment = RuleImpactExperiment(rules)
    experiment.run_experiment()