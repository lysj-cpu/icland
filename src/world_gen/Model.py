import math
import random
from enum import Enum

# You might want this function to emulate the "distribution.Random(randValue)" call
# which selects an index t with probability distribution[t].
def random_index_from_distribution(distribution, rand_value):
    """
    Select an index from 'distribution' proportionally to the values in 'distribution'.
    'rand_value' is a random float in [0,1).
    
    Returns:
        index (int) chosen according to the weights in 'distribution'.
        If the sum of 'distribution' is 0, returns -1 as an error code.
    """
    total = sum(distribution)
    if total <= 0:
        return -1
    cumulative = 0.0
    for i, w in enumerate(distribution):
        cumulative += w
        if rand_value * total <= cumulative:
            return i
    return len(distribution) - 1  # Fallback if floating-point issues occur


class Heuristic(Enum):
    """
    Equivalent to the public enum Heuristic { Entropy, MRV, Scanline } in C#.
    """
    ENTROPY = 1
    MRV = 2
    SCANLINE = 3


class Model:
    """
    Python translation of the C# abstract class 'Model' with added comments.
    """

    # Class-wide (static) arrays, analogous to the protected static int[] in C#
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    opposite = [2, 3, 0, 1]  # Opposite directions for 0->2, 1->3, 2->0, 3->1

    def __init__(self, width, height, N, periodic, heuristic):
        """
        Translated constructor from C#:
        protected Model(int width, int height, int N, bool periodic, Heuristic heuristic)
        """
        self.MX = width
        self.MY = height
        self.N = N
        self.periodic = periodic
        self.heuristic = heuristic

        # Additional fields based on the original code (some remain uninitialized here until init())
        self.wave = None
        self.compatible = None
        self.propagator = None
        self.weights = None
        self.weightLogWeights = None
        self.distribution = None
        self.observed = None
        self.stack = None
        self.stacksize = 0
        self.observedSoFar = 0

        self.T = 0       # number of possible tile/pattern indices (must be set by child or otherwise)
        self.ground = False

        self.sumsOfOnes = None
        self.sumsOfWeights = None
        self.sumsOfWeightLogWeights = None
        self.entropies = None

        self.sumOfWeights = 0.0
        self.sumOfWeightLogWeights = 0.0
        self.startingEntropy = 0.0

    def init(self):
        """
        Equivalent to the void Init() method in the C# code.
        Initializes arrays and precomputes weight sums/log sums.
        """
        # 'wave' is a 2D array [MX*MY][T] in C#
        self.wave = [[False] * self.T for _ in range(self.MX * self.MY)]
        # 'compatible' is a 3D array [MX*MY][T][4]
        self.compatible = [[[0]*4 for _ in range(self.T)] for _ in range(self.MX * self.MY)]
        
        # fill 'compatible' with 0 for now; it gets updated in 'Clear'
        # 'distribution' is for storing weights of active patterns in observe()
        self.distribution = [0.0] * self.T

        # 'observed' is the final chosen pattern index at each cell
        self.observed = [-1] * (self.MX * self.MY)

        # Precompute weightLogWeights, sumOfWeights, sumOfWeightLogWeights
        self.weightLogWeights = [0.0] * self.T
        self.sumOfWeights = 0.0
        self.sumOfWeightLogWeights = 0.0
        
        for t in range(self.T):
            self.weightLogWeights[t] = self.weights[t] * math.log(self.weights[t])
            self.sumOfWeights += self.weights[t]
            self.sumOfWeightLogWeights += self.weightLogWeights[t]

        self.startingEntropy = math.log(self.sumOfWeights) - (self.sumOfWeightLogWeights / self.sumOfWeights)

        # sumsOfOnes, sumsOfWeights, sumsOfWeightLogWeights, entropies track wave state
        self.sumsOfOnes = [0] * (self.MX * self.MY)
        self.sumsOfWeights = [0.0] * (self.MX * self.MY)
        self.sumsOfWeightLogWeights = [0.0] * (self.MX * self.MY)
        self.entropies = [0.0] * (self.MX * self.MY)

        # stack is an array of (int i, int t)
        # In Python, we can just keep a list of tuples.
        self.stack = []
        self.stacksize = 0

    def run(self, seed, limit):
        """
        Translated from public bool Run(int seed, int limit).
        Attempts to observe and propagate patterns up to 'limit' iterations (or indefinitely if limit < 0).
        
        Returns:
            True if a valid tiling is completed or still uncontradicted after limit steps.
            False if a contradiction is found (propagation fails).
        """
        if self.wave is None:
            self.init()

        self.clear()  # Reset wave, etc.
        
        rng = random.Random(seed)  # The C# code uses 'new Random(seed)'

        # For loop for up to 'limit' steps, or infinite if limit < 0
        steps = 0
        while True:
            if limit >= 0 and steps >= limit:
                # If we've reached the limit, just return True
                # (the original code says "for (int l=0; l<limit || limit<0; l++)")
                break
            steps += 1

            node = self.next_unobserved_node(rng)
            if node >= 0:
                # There's an unobserved node, so observe and propagate
                self.observe(node, rng)
                success = self.propagate()
                if not success:
                    return False
            else:
                # No unobserved node found => wave is fully observed
                for i in range(len(self.wave)):
                    for t in range(self.T):
                        if self.wave[i][t]:
                            self.observed[i] = t
                            break
                return True
        
        # If we exit the loop (limit reached without contradiction), return True
        return True

    def next_unobserved_node(self, rng):
        """
        Translated from int NextUnobservedNode(Random random).
        Selects the next cell to observe according to the chosen heuristic (Scanline, Entropy, or MRV).
        
        Returns:
            index (int) of the chosen cell in [0, MX*MY), or -1 if all cells are determined.
        """
        # Handle the SCANLINE heuristic
        if self.heuristic == Heuristic.SCANLINE:
            for i in range(self.observedSoFar, len(self.wave)):
                # skip if out of range (non-periodic boundary constraints)
                if (not self.periodic) and ((i % self.MX) + self.N > self.MX or (i // self.MX) + self.N > self.MY):
                    continue
                if self.sumsOfOnes[i] > 1:
                    self.observedSoFar = i + 1
                    return i
            return -1

        # Handle ENTROPY or MRV
        min_entropy = 1e4
        argmin = -1
        for i in range(len(self.wave)):
            # skip out-of-range if non-periodic
            if (not self.periodic) and ((i % self.MX) + self.N > self.MX or (i // self.MX) + self.N > self.MY):
                continue

            remaining_values = self.sumsOfOnes[i]

            # ENTROPY -> we look at entropies[i], MRV -> we look at remaining_values
            if self.heuristic == Heuristic.ENTROPY:
                entropy = self.entropies[i]
            else:  # MRV
                entropy = float(remaining_values)

            if remaining_values > 1 and entropy <= min_entropy:
                # small noise to break ties
                noise = 1e-6 * rng.random()
                if entropy + noise < min_entropy:
                    min_entropy = entropy + noise
                    argmin = i
        return argmin

    def observe(self, node, rng):
        """
        Translated from void Observe(int node, Random random).
        Collapses the wave at 'node' by picking a pattern index according to weights distribution.
        Then bans all other patterns at that node.
        """
        w = self.wave[node]
        
        # Prepare distribution of patterns that are still possible
        for t in range(self.T):
            self.distribution[t] = self.weights[t] if w[t] else 0.0
        
        r = random_index_from_distribution(self.distribution, rng.random())

        # Ban any pattern that isn't the chosen one
        for t in range(self.T):
            # If wave[node][t] != (t == r) => ban it
            if w[t] != (t == r):
                self.ban(node, t)

    def propagate(self):
        """
        Translated from bool Propagate().
        
        Returns:
            True if propagation completed successfully (no contradictions found).
            (In the original snippet, it returns sumsOfOnes[0] > 0, which is suspicious,
            but we'll keep the same logic.)
        """
        while self.stacksize > 0:
            i1, t1 = self.stack[self.stacksize - 1]
            self.stacksize -= 1

            x1 = i1 % self.MX
            y1 = i1 // self.MX

            for d in range(4):
                x2 = x1 + self.dx[d]
                y2 = y1 + self.dy[d]

                # Boundary checks if not periodic
                if (not self.periodic) and (x2 < 0 or y2 < 0 or x2 + self.N > self.MX or y2 + self.N > self.MY):
                    continue

                # Wrap around if periodic
                if x2 < 0:
                    x2 += self.MX
                elif x2 >= self.MX:
                    x2 -= self.MX
                if y2 < 0:
                    y2 += self.MY
                elif y2 >= self.MY:
                    y2 -= self.MY

                i2 = x2 + y2 * self.MX
                p = self.propagator[d][t1]  # Patterns that can follow t1 in direction d
                compat = self.compatible[i2]

                # For each pattern t2 that can follow t1, decrease its compatibility count
                for t2 in p:
                    comp = compat[t2]
                    comp[d] -= 1
                    if comp[d] == 0:
                        # If no compatibility remains in direction d, ban this pattern
                        self.ban(i2, t2)

        # The original snippet returns sumsOfOnes[0] > 0.
        # This is not typical for full WFC, but we'll preserve it.
        return self.sumsOfOnes[0] > 0

    def ban(self, i, t):
        """
        Translated from void Ban(int i, int t).
        Bans pattern t at cell i. Updates wave, compatibility, sumsOfOnes, entropies, and stack.
        """
        # If wave[i][t] is already false, do nothing
        if not self.wave[i][t]:
            return

        self.wave[i][t] = False

        # Zero-out the compatibility in all directions for pattern t at cell i
        comp = self.compatible[i][t]
        for d in range(4):
            comp[d] = 0

        # Push (i, t) onto stack
        if self.stacksize < len(self.stack):
            self.stack[self.stacksize] = (i, t)
        else:
            self.stack.append((i, t))
        self.stacksize += 1

        # Update sumsOfOnes, sumsOfWeights, sumsOfWeightLogWeights, entropies
        self.sumsOfOnes[i] -= 1
        self.sumsOfWeights[i] -= self.weights[t]
        self.sumsOfWeightLogWeights[i] -= self.weightLogWeights[t]

        sum_w = self.sumsOfWeights[i]
        self.entropies[i] = math.log(sum_w) - (self.sumsOfWeightLogWeights[i] / sum_w) if sum_w > 0 else 0.0

    def clear(self):
        """
        Translated from void Clear().
        Resets wave and compatibility to allow all patterns at all cells,
        then optionally applies 'ground' constraints.
        """
        for i in range(len(self.wave)):
            for t in range(self.T):
                self.wave[i][t] = True
                # Set each direction's compatibility count to the number of patterns
                # that can appear in the cell adjacent in that direction.
                # The code does: compatible[i][t][d] = propagator[opposite[d]][t].Length
                for d in range(4):
                    # Because we do not show how 'propagator' is fully set up,
                    # we assume it is an array-of-arrays of pattern-lists.
                    self.compatible[i][t][d] = len(self.propagator[self.opposite[d]][t])

            self.sumsOfOnes[i] = len(self.weights)
            self.sumsOfWeights[i] = self.sumOfWeights
            self.sumsOfWeightLogWeights[i] = self.sumOfWeightLogWeights
            self.entropies[i] = self.startingEntropy
            self.observed[i] = -1

        self.observedSoFar = 0

        if self.ground:
            # Force the bottom row to be the "ground" pattern
            for x in range(self.MX):
                # Ban all patterns except the last one (T-1) in the bottom row
                for t in range(self.T - 1):
                    self.ban(x + (self.MY - 1) * self.MX, t)
                # For other rows above the bottom, ban the ground pattern
                for y in range(self.MY - 1):
                    self.ban(x + y * self.MX, self.T - 1)

            # Propagate after applying ground constraints
            self.propagate()

    def save(self, filename):
        """
        Translated from the abstract public void Save(string filename).
        Since this is an abstract class in C#, we simply raise NotImplementedError in Python.
        """
        raise NotImplementedError("Must be implemented in a subclass.")
