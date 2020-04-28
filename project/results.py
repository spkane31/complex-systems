import copy
import datetime
import json

class RunResults():
    def __init__(self):
        self._d = {}
        self._d["datetime"] = datetime.datetime.now().isoformat(sep=" ")
    
    def __getitem__(self, key):
        return self._d[key]
    
    def as_dict(self):
        """
        Return a copy of the internal state of the `SwarmResults` as a `dict`.
        """
        return copy.deepcopy(self._d)
    
    def save_json(self, file_path: str):
        """
        Dumps the `SwarmResults` to a provided file path in JSON format.

        Warning: This method will overwrite the provided file if it already
        exists without prompting!

        file_path (str): The destination file path for the JSON output.
        """
        with open(file_path, "w") as jsonf:
            json.dump(self._d, jsonf)
    
    # epochs is a property; it can be used just like if it were a member field:
    # >>> s = RunResults()
    # >>> s.epochs += 1
    # >>> s.epochs
    # 1
    # >>> if s.epochs < 3:
    # >>>   print("epochs is low")
    # epochs is low
    # >>> 
    @property
    def epochs(self):
        return self._d.get("epochs", 0)
    @epochs.setter
    def epochs(self, new_epochs):
        self._d["epochs"] = new_epochs
    @epochs.deleter
    def epochs(self):
        self._d.pop("epochs", None)
    
    def _add(self, key, epoch, items, _initclass=dict):
        if key not in self._d:
            self._d[key] = _initclass()
        self._d[key][epoch] = items
    
    def add_pso_params(self, fitness_function, dimension, w, C, S, **kwargs):
        pso_params = {
            "dimension": dimension,
            "fitness_function": fitness_function,
            "w": w,
            "C": C,
            "S": S
        }
        pso_params.update(kwargs)
        self._d["pso_params"] = pso_params
    
    def add_scores(self, epoch, scores):
        """
        Add the scores at a certain epoch to the results log.

        epoch (int): An integer denoting the epoch of the provided scores.

        scores (List[float]): A list of scores where each entry corresponds to
        a single particle's score (fitness at current position).
        """
        self._add("scores", epoch, copy.deepcopy(scores))
    
    def add_positions(self, epoch, particles):
        """
        Add the particle positions at a certain epoch to the results log.

        epoch (int): An integer denoting the epoch of the provided positions.

        particles (List[Particle]): A list of `Particle`s to take the current
        position from.
        """
        self._add(
            "positions",
            epoch,
            [copy.deepcopy(p.currentPos) for p in particles]
        )

    def add_particle_bests(self, epoch, particles):
        """
        Add the particle self-bests at a certain epoch to the results log.

        epoch (int): An integer denoting the epoch of the provided self-bests.

        particles (List[Particle]): A list of `Particle`s to take the self
        best positions from.
        """
        self._add(
            "self_bests",
            epoch,
            [copy.deepcopy(p.selfBest) for p in particles]
        )
    
    def add_global_best(self, epoch, global_best):
        """
        Add the swarm global best location at a certain epoch to the results
        log.

        epoch (int): An integer denoting the epoch of the provided global best.

        global_best (List[float]): The position of the swarm global best.
        """
        self._add("global_bests", epoch, copy.deepcopy(global_best))

    def add_runtime(self, runtime):
        self._d["runtime"] = "{:.3f}".format(runtime)
