import math
import xml.etree.ElementTree as ET
import os
from collections import defaultdict
from enum import Enum
from Model import Model

from PIL import Image

def load_bitmap(filepath):
    """
    Loads an image file (e.g., PNG) into a list of ARGB-encoded 32-bit integers.
    
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

        # Convert each RGBA tuple into a single ARGB integer
        # A in bits 24..31, R in bits 16..23, G in bits 8..15, B in bits 0..7
        result = []
        for (r, g, b, a) in pixel_data:
            argb = (a << 24) | (r << 16) | (g << 8) | b
            result.append(argb)
        
    return (result, width, height)


def save_bitmap(data, width, height, filename):
    """
    Saves a list of ARGB-encoded 32-bit integers as an image file (e.g., PNG).
    
    Arguments:
        data: a list of length width*height containing 0xAARRGGBB pixels
        width, height: image dimensions
        filename: path to save the resulting image file
    """
    # Create a new RGBA image
    img = Image.new("RGBA", (width, height))
    
    # Convert each ARGB int back into an (R, G, B, A) tuple
    rgba_pixels = []
    for argb in data:
        a = (argb >> 24) & 0xFF
        r = (argb >> 16) & 0xFF
        g = (argb >> 8)  & 0xFF
        b = (argb >> 0)  & 0xFF
        rgba_pixels.append((r, g, b, a))
    
    # Put these pixels into the image and save
    img.putdata(rgba_pixels)
    img.save(filename, format="PNG")

# Helper to get XML attribute with a default (similar to xelem.Get<T>(...))
def get_xml_attribute(xelem, attribute, default=None, cast_type=None):
    """
    Returns the value of 'attribute' from the XML element xelem.
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


class SimpleTiledModel(Model):
    """
    Python translation of the C# SimpleTiledModel class which inherits from Model.
    """

    def __init__(self, name, subsetName, width, height, periodic, blackBackground, heuristic):
        """
        Constructor, translated from:
          public SimpleTiledModel(string name, string subsetName, int width, int height,
                                  bool periodic, bool blackBackground, Heuristic heuristic)
        Calls the base Model constructor with N=1 (since SimpleTiledModel is typically single-tile WFC).
        """
        super().__init__(width, height, N=1, periodic=periodic, heuristic=heuristic)

        self.blackBackground = blackBackground
        self.tiles = []         # Will hold arrays of pixel data for each tile variant
        self.tilenames = []     # Will hold tile names (including variants)
        self.tilesize = 0       # Size (width==height) of each tile in pixels

        # Load XML file: "tilesets/<name>.xml"
        xml_path = os.path.join("images/samples", name, "data.xml")
        tree = ET.parse(xml_path)
        xroot = tree.getroot()

        # Read whether tiles are "unique"
        unique = get_xml_attribute(xroot, "unique", default=False, cast_type=bool)

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
        def tile(f, size):
            """
            Creates a flat list of length size*size by calling f(x,y) for each pixel.
            """
            result = [0] * (size * size)
            for y in range(size):
                for x in range(size):
                    result[x + y * size] = f(x, y)
            return result

        def rotate(array, size):
            """
            Rotates the array by 90 degrees clockwise.
            The function is: new[x,y] = old[size-1-y, x].
            """
            return tile(lambda x, y: array[size - 1 - y + x * size], size)

        def reflect(array, size):
            """
            Reflects (mirror) the array horizontally.
            The function is: new[x,y] = old[size-1-x, y].
            """
            return tile(lambda x, y: array[size - 1 - x + y * size], size)

        # We'll maintain a list of transformations (the 'action' array in C#).
        # In Python, we'll call it `actions`. Each item is an array of 8 transformations.
        actions = []
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
            sym = get_xml_attribute(xtile, "symmetry", default='X', cast_type=str)
            w = get_xml_attribute(xtile, "weight", default=1.0, cast_type=float)

            # Determine the group transformations: cardinality, rotation function 'a', reflection function 'b'
            if sym == 'L':
                cardinality = 4
                a = lambda i: (i + 1) % 4
                b = lambda i: i + 1 if (i % 2 == 0) else i - 1
            elif sym == 'T':
                cardinality = 4
                a = lambda i: (i + 1) % 4
                b = lambda i: i if (i % 2 == 0) else 4 - i
            elif sym == 'I':
                cardinality = 2
                a = lambda i: 1 - i
                b = lambda i: i
            elif sym == '\\':
                cardinality = 2
                a = lambda i: 1 - i
                b = lambda i: 1 - i
            elif sym == 'F':
                cardinality = 8
                a = lambda i: (i + 1) % 4 if i < 4 else 4 + ((i - 1) % 4)
                b = lambda i: i + 4 if i < 4 else i - 4
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
            # The original code checks "unique" to see if we have separate PNGs for each rotation.
            if unique:
                # Each orientation is stored in a separate file, e.g. "<tilename> 0.png", "<tilename> 1.png", etc.
                for t in range(cardinality):
                    bitmap, w_img, h_img = load_bitmap(os.path.join("images/samples", name, f"{tilename} {t}.png"))
                    # Usually, w_img == h_img => tile is square
                    if self.tilesize == 0:
                        self.tilesize = w_img
                    self.tiles.append(bitmap)
                    self.tilenames.append(f"{tilename} {t}")
            else:
                # Single PNG for the base tile
                bitmap, w_img, h_img = load_bitmap(os.path.join("images/samples", name, f"{tilename}.png"))
                if self.tilesize == 0:
                    self.tilesize = w_img
                base_idx = len(self.tiles)
                self.tiles.append(bitmap)
                self.tilenames.append(f"{tilename} 0")

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
                        self.tiles[-1] = reflected if t <= 3 else self.tiles[-1]  # Adjust if needed
                        # Actually, we should do a separate entry:
                        self.tiles.append(reflected)
                    self.tilenames.append(f"{tilename} {t}")

            # Weighted for each orientation
            for _ in range(cardinality):
                weightList.append(w)

        # The total number of distinct tile variants T is the final length of `actions`.
        self.T = len(actions)
        # Convert weightList to a python list of floats
        self.weights = [float(x) for x in weightList]

        # Build the propagator arrays: self.propagator[d][t] = list of tile indices that can appear
        # in direction d next to tile t.
        # We'll do a 3D structure: [4][T][variable-size list], same as in the C# code.
        self.propagator = [[[] for _ in range(self.T)] for _ in range(4)]

        # We'll build a "densePropagator[d][t1][t2] = True/False" for adjacency, then convert
        # to a sparse list of valid t2's for each t1.
        densePropagator = [[[False for _ in range(self.T)] for _ in range(self.T)] for _ in range(4)]

        # Parse neighbors from the <neighbors> section
        neighbors_elem = xroot.find("neighbors")
        if neighbors_elem:
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
                    print(f"ERROR: tile {self.tilenames[t1]} has no neighbors in direction {d}")

                self.propagator[d][t1] = valid_t2s

    def save(self, filename):
        """
        Override of the Model.save() method.
        Translated from:
          public override void Save(string filename)
        """
        # We'll create a pixel buffer for the entire output image:
        # (MX * tilesize) by (MY * tilesize).
        bitmapData = [0] * (self.MX * self.MY * self.tilesize * self.tilesize)

        # If we have a definite observation (observed[0]>=0 means not contradictory)
        if self.observed[0] >= 0:
            # For each cell (x,y), pick the corresponding tile's pixel data
            for y in range(self.MY):
                for x in range(self.MX):
                    tile_index = self.observed[x + y * self.MX]
                    tile_data = self.tiles[tile_index]

                    for dy in range(self.tilesize):
                        for dx in range(self.tilesize):
                            sx = x * self.tilesize + dx
                            sy = y * self.tilesize + dy
                            bitmapData[sx + sy * (self.MX * self.tilesize)] = tile_data[dx + dy * self.tilesize]
        else:
            # Not fully observed -> show "superposition" or black background
            for i in range(len(self.wave)):
                x = i % self.MX
                y = i // self.MX

                if self.blackBackground and self.sumsOfOnes[i] == self.T:
                    # Paint as black (255 << 24 => 0xff000000 in ARGB)
                    for yt in range(self.tilesize):
                        for xt in range(self.tilesize):
                            sx = x * self.tilesize + xt
                            sy = y * self.tilesize + yt
                            bitmapData[sx + sy * (self.MX * self.tilesize)] = 0xff000000
                else:
                    w = self.wave[i]
                    normalization = 1.0 / self.sumsOfWeights[i] if self.sumsOfWeights[i] != 0 else 0
                    for yt in range(self.tilesize):
                        for xt in range(self.tilesize):
                            sx = x * self.tilesize + xt
                            sy = y * self.tilesize + yt

                            r = g = b = 0.0
                            for t in range(self.T):
                                if w[t]:
                                    argb = self.tiles[t][xt + yt * self.tilesize]
                                    # ARGB channels
                                    rr = (argb & 0xff0000) >> 16
                                    gg = (argb & 0x00ff00) >> 8
                                    bb = (argb & 0x0000ff)
                                    wgt = self.weights[t] * normalization
                                    r += rr * wgt
                                    g += gg * wgt
                                    b += bb * wgt
                            # Combine into ARGB int, alpha=255
                            R = int(r)
                            G = int(g)
                            B = int(b)
                            A = 0xff << 24
                            pixel = A | (R << 16) | (G << 8) | B
                            bitmapData[sx + sy * (self.MX * self.tilesize)] = pixel

        # Finally, save the image
        save_bitmap(bitmapData, self.MX * self.tilesize, self.MY * self.tilesize, filename)

    def text_output(self):
        """
        Translated from:
          public string TextOutput()
        Returns a textual representation of the observed tilenames in each cell row-by-row.
        """
        # If `observed[x + y * MX]` is -1, that means unobserved or uncertain
        lines = []
        for y in range(self.MY):
            row_names = []
            for x in range(self.MX):
                obs_index = self.observed[x + y * self.MX]
                if obs_index >= 0 and obs_index < len(self.tilenames):
                    row_names.append(self.tilenames[obs_index])
                else:
                    row_names.append("???")
            lines.append(", ".join(row_names))
        return "\n".join(lines) + "\n"
