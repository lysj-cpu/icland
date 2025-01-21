"""This file contains the base Model class for WaveFunctionCollapse and helper functions."""
from flax import struct

import math
import random
from enum import Enum

def random_index_from_distribution(distribution, rand_value):
    """Select an index from 'distribution' proportionally to the values in 'distribution'.
    
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
    """Enum for the heuristic selection in WaveFunctionCollapse."""
    ENTROPY = 1
    MRV = 2
    SCANLINE = 3

@struct.dataclass
class Model:
    """Base Model class for WaveFunctionCollapse algorithm."""

    # Class-wide (static) arrays, analogous to the protected static int[] in C#
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    opposite = [2, 3, 0, 1]  # Opposite directions for 0->2, 1->3, 2->0, 3->1

    def __init__(self, width, height, T, j_weights, j_propagator, tilecodes):
        """Constructor for the Model class."""
        self.MX = width
        self.MY = height

        self.T = T
        self.j_weights = j_weights
        self.j_propagator = j_propagator
        self.tilecodes = tilecodes


    def run(self, seed):
        """Run the WaveFunctionCollapse algorithm with the given seed and iteration limit."""
        limit = 10000   # Arbitary number for now

        self.clear()
        
        # TODO: use jit-able random number generator
        rng = random.Random(seed)

        steps = 0
        while True:
            if limit >= 0 and steps >= limit:
                break
            steps += 1

            node = self.next_unobserved_node(rng)
            if node >= 0:
                self.observe(node, rng)
                success = self.propagate()
                if not success:
                    return False
            else:
                for i in range(len(self.wave)):
                    for t in range(self.T):
                        if self.wave[i][t]:
                            self.observed[i] = t
                            break
                return True
        
        return True

    def next_unobserved_node(self, rng):
        """Selects the next cell to observe according to the chosen heuristic (Scanline, Entropy, or MRV).
        
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
        """Collapses the wave at 'node' by picking a pattern index according to weights distribution.

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
        """Propagate the wave function collapse constraints.

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
        """Bans pattern t at cell i. Updates wave, compatibility, sumsOfOnes, entropies, and stack."""
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
        """Resets wave and compatibility to allow all patterns at all cells, then optionally applies 'ground' constraints."""
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
        """Raises NotImplementedError to ensure its subclass will provide an implementation."""
        raise NotImplementedError("Must be implemented in a subclass.")
