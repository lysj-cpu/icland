"""Model for the WaveFunctionCollapse algorithm, JIT-compiled with JAX."""

import os
import xml.etree.ElementTree as ET
from collections.abc import Callable
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from PIL import Image


class Heuristic(Enum):
    """Enum for the heuristic selection in WaveFunctionCollapse."""

    ENTROPY = 1
    MRV = 2
    SCANLINE = 3


class TileType(Enum):
    """Enum for types of tiles used in the tilemap."""

    SQUARE = 0
    RAMP = 1
    VRAMP = 2


def load_bitmap(filepath: str) -> tuple[list[int], int, int]:
    """Loads an image file (e.g., PNG) into a list of ARGB-encoded 32-bit integers.

    Returns:
        (pixels, width, height)
          pixels: a list of length (width*height), each an integer 0xAARRGGBB
          width, height: dimensions of the image
    """
    with Image.open(filepath) as img:
        # Convert the image to RGBA so we have consistent 4-channel data
        img = img.convert("RGBA")
        width, height = img.size

        # Retrieve pixel data in (R, G, B, A) tuples
        pixel_data = list(img.getdata())

        # Convert each RGBA Tuple into a single ARGB integer
        # A in bits 24..31, R in bits 16..23, G in bits 8..15, B in bits 0..7
        result = []
        for r, g, b, a in pixel_data:
            argb = (a << 24) | (r << 16) | (g << 8) | b
            result.append(argb)

    return (result, width, height)


def save_bitmap(data: list[int], width: int, height: int, filename: str) -> None:
    """Saves a list of ARGB-encoded 32-bit integers as an image file (e.g., PNG).

    Arguments:
        data: a list of length width*height containing 0xAARRGGBB pixels
        width: image width
        height: image height
        filename: path to save the resulting image file
    """
    # Create a new RGBA image
    img = Image.new("RGBA", (width, height))

    # Convert each ARGB int back into an (R, G, B, A) Tuple
    rgba_pixels = []
    for argb in data:
        a = (argb >> 24) & 0xFF
        r = (argb >> 16) & 0xFF
        g = (argb >> 8) & 0xFF
        b = (argb >> 0) & 0xFF
        rgba_pixels.append((r, g, b, a))

    # Put these pixels into the image and save
    img.putdata(rgba_pixels)
    img.save(filename, format="PNG")


# Helper to get XML attribute with a default (similar to xelem.Get<T>(...))
def get_xml_attribute(
    xelem: ET.Element, attribute: str, default: Any = None, cast_type: Any = None
) -> Any:
    """Returns the value of 'attribute' from the XML element xelem.

    If the attribute is not present, returns 'default'.
    If cast_type is not None, attempts to cast the attribute's string value to that type.
    """
    val = xelem.get(attribute)
    if val is None:
        return default
    if cast_type:
        if cast_type is bool:
            # In XML, might be "true"/"false", we handle that
            return val.lower() in ("true", "1")
        elif cast_type is int:
            return int(val)
        elif cast_type is float:
            return float(val)
        else:
            # Attempt a generic constructor cast
            return cast_type(val)
    return val


class XMLReader:
    """A class to parse tilemap XML files and handle tile-related data for WaveFunctionCollapse.

    This class reads tile data, processes their transformations (rotations, reflections),
    and prepares metadata for use in tile-based procedural generation.

    Attributes:
        tiles (list): Pixel data arrays for each tile variant.
        tilenames (list): Names of tiles (including variants).
        tilesize (int): Size (width and height) of each tile in pixels.
        tilecodes (list): Encoded tile properties as 4-tuples (type, rotation, from, to).
        weights (list): Weights associated with each tile variant.
        propagator (list): Sparse adjacency data indicating valid neighboring tiles.
        j_propagator (jax.numpy.array): JAX-compatible array representation of `propagator`.
        j_weights (jax.numpy.array): JAX-compatible array of tile weights.
        j_tilecodes (jax.numpy.array): JAX-compatible array of tile properties.
    """

    def __tilename_to_code(self, tile: str, rotation: int) -> tuple[int, int, int, int]:
        name_split = tile.split("_")
        tile_type = name_split[0]
        tile_type_num = 0

        from_num = 0
        to_num = 0

        is_ramp_type = False
        if tile_type == "square":
            tile_type_num = TileType.SQUARE.value
        elif tile_type == "ramp":
            tile_type_num = TileType.RAMP.value
            is_ramp_type = True
        elif tile_type == "vramp":
            tile_type_num = TileType.VRAMP.value
            is_ramp_type = True
        else:
            raise RuntimeError("Unknown tile type.")

        if is_ramp_type:
            from_num, to_num = int(name_split[1]), int(name_split[2])
        else:
            to_num = int(name_split[-1])

        return tile_type_num, rotation, from_num, to_num

    def __init__(self, xml_path: str, subsetName: (str | None) = None) -> None:
        """Initializes the XMLReader by parsing the tilemap XML file.

        Argumentss:
            subsetName (str, optional): Name of a specific subset of tiles to load.
        """
        # Load XML file: "tilesets/<name>.xml"
        tree = ET.parse(xml_path)
        xroot = tree.getroot()

        self.tiles: list[
            list[int]
        ] = []  # Will hold arrays of pixel data for each tile variant
        self.tilenames: list[str] = []  # Will hold tile names (including variants)
        self.tilesize: int = 0  # Size (width==height) of each tile in pixels

        self.tilecodes = []  # Will hold 4-Tuple of (type, rotation, from, to)

        # Prepare optional subset
        subset = None
        if subsetName is not None:
            # <subsets><subset name="..."><tile name="..."/></subset></subsets>
            # We find the correct <subset> with name == subsetName
            xsubsets = xroot.find("subsets")
            if xsubsets is not None:
                xsubset = None
                for elem in xsubsets.findall("subset"):
                    # Check if <subset name="subsetName">
                    n = get_xml_attribute(elem, "name")
                    if n == subsetName:
                        xsubset = elem
                        break
                if xsubset is not None:
                    # Gather tile names in this subset
                    subset = []
                    for tile_elem in xsubset.findall("tile"):
                        tile_name = get_xml_attribute(tile_elem, "name", cast_type=str)
                        if tile_name is not None:
                            subset.append(tile_name)
                else:
                    print(f"ERROR: subset {subsetName} not found.")
            else:
                print(f"ERROR: <subsets> not found in {xml_path}.")

        # Local helper functions to rotate and reflect pixel arrays
        def tile(f: Callable[[int, int], int], size: int) -> list[int]:
            """Creates a flat list of length size*size by calling f(x,y) for each pixel."""
            result = [0] * (size * size)
            for y in range(size):
                for x in range(size):
                    result[x + y * size] = f(x, y)
            return result

        def rotate(array: list[int], size: int) -> list[int]:
            """Rotates the array by 90 degrees clockwise.

            The function is: new[x,y] = old[size-1-y, x].
            """
            return tile(lambda x, y: array[size - 1 - y + x * size], size)

        def reflect(array: list[int], size: int) -> list[int]:
            """Reflects (mirror) the array horizontally.

            The function is: new[x,y] = old[size-1-x, y].
            """
            return tile(lambda x, y: array[size - 1 - x + y * size], size)

        # We'll maintain a list of transformations (the 'action' array in C#).
        # In Python, we'll call it `actions`. Each item is an array of 8 transformations.
        actions: list[list[int]] = []
        firstOccurrence = {}

        # We'll accumulate weights in a list, then convert to a Python list or NumPy array later.
        weightList = []

        # <tiles><tile name="..." symmetry="..." weight="..."/></tiles>
        tiles_elem = xroot.find("tiles")
        if tiles_elem is None:
            raise ValueError(f"XML file {xml_path} missing <tiles> section.")

        for xtile in tiles_elem.findall("tile"):
            tilename = get_xml_attribute(xtile, "name", cast_type=str)
            if tilename is None:
                continue

            # If there's a subset, and this tile isn't in it, skip
            if subset is not None and tilename not in subset:
                continue

            # Read tile's symmetry (default 'X' if not present)
            sym = get_xml_attribute(xtile, "symmetry", default="X", cast_type=str)
            w = get_xml_attribute(xtile, "weight", default=1.0, cast_type=float)

            # Determine the group transformations: cardinality, rotation function 'a', reflection function 'b'
            if sym == "L":
                cardinality = 4
                a: Callable[[int], int] = lambda i: (i + 1) % 4
                b: Callable[[int], int] = lambda i: i + 1 if (i % 2 == 0) else i - 1
            elif sym == "T":
                cardinality = 4
                a = lambda i: (i + 1) % 4
                b = lambda i: i if (i % 2 == 0) else 4 - i
            else:
                # 'X' or any unspecified
                cardinality = 1
                a = lambda i: i
                b = lambda i: i

            # In the original code, T = action.Count
            # Because we're adding a block of transformations for this tile,
            # we store that starting index in 'Tstart'.
            Tstart = len(actions)

            # Save the first occurrence of this tile name
            # (We assume each tile name is unique across the entire tileset.)
            firstOccurrence[tilename] = Tstart

            # For each 't' in [0..cardinality), build the 8 transformations: [0..7]
            # map[t][s], which is a re-labeling among the T expansions.
            # We'll collect these in an array for the tile's transformations.
            # Then we add them to `actions`.
            map_t = []
            for t in range(cardinality):
                # array of length 8
                row = [0] * 8
                row[0] = t
                row[1] = a(t)
                row[2] = a(a(t))
                row[3] = a(a(a(t)))
                row[4] = b(t)
                row[5] = b(a(t))
                row[6] = b(a(a(t)))
                row[7] = b(a(a(a(t))))

                # Then we offset all by Tstart so each transformation is unique in the global indexing
                for s in range(8):
                    row[s] += Tstart

                # We'll add 'row' to map_t
                map_t.append(row)

            # Now that we have map_t, we push each row into `actions`.
            for t in range(cardinality):
                actions.append(map_t[t])

            # Next, we load the actual pixel data.
            # Single PNG for the base tiles
            bitmap, w_img, h_img = load_bitmap(
                os.path.abspath(
                    os.path.join(os.path.join(xml_path, os.pardir), f"{tilename}.png")
                )
            )
            if self.tilesize == 0:
                self.tilesize = w_img
            base_idx = len(self.tiles)
            self.tiles.append(bitmap)
            self.tilenames.append(f"{tilename} 0")

            self.tilecodes.append(self.__tilename_to_code(tilename, 0))

            # Then produce the rest by rotate/reflect in code if cardinality > 1
            for t in range(1, cardinality):
                # If t <= 3 => rotate previous tile
                if t <= 3:
                    rotated = rotate(self.tiles[base_idx + t - 1], self.tilesize)
                    self.tiles.append(rotated)
                # If t >= 4 => reflect tile [base_idx + t - 4]
                if t >= 4:
                    reflected = reflect(self.tiles[base_idx + t - 4], self.tilesize)
                    # Overwrite the just-added tile, or add a new entry
                    self.tiles[-1] = (
                        reflected if t <= 3 else self.tiles[-1]
                    )  # Adjust if needed
                    # Actually, we should do a separate entry:
                    self.tiles.append(reflected)
                self.tilenames.append(f"{tilename} {t}")

                self.tilecodes.append(self.__tilename_to_code(tilename, t))

            # Weighted for each orientation
            for _ in range(cardinality):
                weightList.append(w)

        # The total number of distinct tile variants T is the final length of `actions`.
        self.T = len(actions)  # !
        # Convert weightList to a python list of floats
        self.weights = [float(x) for x in weightList]  # !

        # Build the propagator arrays: self.propagator[d][t] = list of tile indices that can appear
        # in direction d next to tile t.
        # We'll do a 3D structure: [4][T][variable-size list], same as in the C# code.
        self.propagator: list[list[list[int]]] = [
            [[] for _ in range(self.T)] for _ in range(4)
        ]  # !

        # We'll build a "densePropagator[d][t1][t2] = True/False" for adjacency, then convert
        # to a sparse list of valid t2's for each t1.
        densePropagator = [
            [[False for _ in range(self.T)] for _ in range(self.T)] for _ in range(4)
        ]

        # Parse neighbors from the <neighbors> section
        neighbors_elem = xroot.find("neighbors")
        if neighbors_elem is not None:
            for xneighbor in neighbors_elem.findall("neighbor"):
                left_str = get_xml_attribute(xneighbor, "left", cast_type=str)
                right_str = get_xml_attribute(xneighbor, "right", cast_type=str)
                if not left_str or not right_str:
                    continue

                # left_str might look like "TileName" or "TileName X"
                left_parts = left_str.split()
                right_parts = right_str.split()

                # If we have a subset, skip if these tiles aren't in it
                if subset is not None:
                    if left_parts[0] not in subset or right_parts[0] not in subset:
                        continue

                left_tile_idx = firstOccurrence[left_parts[0]]
                left_variant = int(left_parts[1]) if len(left_parts) > 1 else 0
                L = actions[left_tile_idx][left_variant]
                D = actions[L][1]  # same as action[L][1] in C#

                right_tile_idx = firstOccurrence[right_parts[0]]
                right_variant = int(right_parts[1]) if len(right_parts) > 1 else 0
                R = actions[right_tile_idx][right_variant]
                U = actions[R][1]  # same as action[R][1]

                # Now set the adjacency in densePropagator
                # direction 0 => left-right adjacency
                densePropagator[0][R][L] = True
                densePropagator[0][actions[R][6]][actions[L][6]] = True
                densePropagator[0][actions[L][4]][actions[R][4]] = True
                densePropagator[0][actions[L][2]][actions[R][2]] = True

                # direction 1 => up-down adjacency
                densePropagator[1][U][D] = True
                densePropagator[1][actions[D][6]][actions[U][6]] = True
                densePropagator[1][actions[U][4]][actions[D][4]] = True
                densePropagator[1][actions[D][2]][actions[U][2]] = True

        # Fill in directions 2,3 as the reverse of 0,1
        # direction 2 => the opposite of 0
        # direction 3 => the opposite of 1
        for t2 in range(self.T):
            for t1 in range(self.T):
                densePropagator[2][t2][t1] = densePropagator[0][t1][t2]
                densePropagator[3][t2][t1] = densePropagator[1][t1][t2]

        # Convert densePropagator to a sparse list in self.propagator
        for d in range(4):
            for t1 in range(self.T):
                valid_t2s = []
                for t2 in range(self.T):
                    if densePropagator[d][t1][t2]:
                        valid_t2s.append(t2)
                if len(valid_t2s) == 0:
                    print(
                        f"ERROR: tile {self.tilenames[t1]} has no neighbors in direction {d}"
                    )

                self.propagator[d][t1] = valid_t2s

        max_len = 0
        for i in range(len(self.propagator)):
            for j in range(len(self.propagator[i])):
                max_len = max(max_len, len(self.propagator[i][j]))

        # Initialize the result array with zeros of shape (4, T, max_len)
        j_prop_result = jnp.zeros((4, self.T, max_len))

        # Function to pad each sequence to max_len
        def pad_sequence(seq: list[list[int]], max_len: int) -> NDArray[np.int64]:
            # Create a result array of shape (T, max_len)
            result = np.full((self.T, max_len), -1)  # Use NumPy array for padding

            # Iterate over each time step (T dimension)
            for i in range(len(seq)):
                seq_len = len(seq[i])
                # Only copy up to the sequence length (no truncation, just padding)
                result[i, :seq_len] = seq[i]

            return result

        padded_sequences = [pad_sequence(seq, max_len) for seq in self.propagator]

        # Convert the padded sequences into a JAX array
        j_padded_seq = jnp.array(padded_sequences)

        # Now assign the padded sequences into the result array
        self.j_propagator = j_prop_result.at[:, :, :].set(j_padded_seq)

        # Now handle weights conversion to a JAX array
        self.j_weights = jnp.array(self.weights)

        self.j_tilecodes = jnp.array(self.tilecodes)

    def save(self, observed: jax.Array, width: int, height: int, filename: str) -> None:
        """Save the tilemap as a bitmap representation for debugging."""
        # We'll create a pixel buffer for the entire output image:
        # (MX * tilesize) by (MY * tilesize).
        tilemap_width, tilemap_height = width, height
        bitmapData = [0] * (
            tilemap_width * tilemap_height * self.tilesize * self.tilesize
        )
        # If we have a definite observation (observed[0]>=0 means not contradictory)
        for y in range(tilemap_height):
            for x in range(tilemap_width):
                # tile index !!
                tile_index = observed.at[x + y * tilemap_width].get()
                tile_data = self.tiles[tile_index]

                for dy in range(self.tilesize):
                    for dx in range(self.tilesize):
                        sx = x * self.tilesize + dx
                        sy = y * self.tilesize + dy
                        bitmapData[sx + sy * (tilemap_width * self.tilesize)] = (
                            tile_data[dx + dy * self.tilesize]
                        )

        # Finally, save the image
        save_bitmap(
            bitmapData,
            tilemap_width * self.tilesize,
            tilemap_height * self.tilesize,
            filename,
        )

    def get_tilemap_data(self) -> tuple[int, jax.Array, jax.Array, jax.Array]:
        """Returns relevant tilemap data.

        Returns:
            tuple: (T, j_weights, j_propagator, j_tilecodes)
                T (int): Number of unique tile variants.
                j_weights (jax.numpy.array): Array of tile weights.
                j_propagator (jax.numpy.array): Propagator data.
                j_tilecodes (jax.numpy.array): Encoded tile properties.
        """
        return self.T, self.j_weights, self.j_propagator, self.j_tilecodes
