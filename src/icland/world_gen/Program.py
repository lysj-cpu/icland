"""Main entry point for the world generation program."""
import os
import glob
import time
import random
import xml.etree.ElementTree as ET
import Model
import SimpleTiledModel
from JITModel import XMLReader

# Assuming you've already implemented or imported the following from your translations:
# from model import Model, Heuristic
# from overlapping_model import OverlappingModel
# from simpletiled_model import SimpleTiledModel
#
# If you used different module or class names, adjust as necessary.


def main():
    """Main entry point for the world generation program."""
    start = time.time()  # Start measuring elapsed time

    # Create "output" directory and delete existing files there.
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    for fpath in glob.glob(os.path.join(out_dir, "*")):
        os.remove(fpath)

    rng = random.Random()

    # Load samples.xml
    tree = ET.parse("samples_reference.xml")
    root = tree.getroot()

    jit_model = XMLReader()
    T, j_weights, j_propagator, tilecodes = jit_model.get_tilemap_data()
    

    # Helper: parse string -> Model.Heuristic
    # (Adjust if your Heuristic enum has different names or structure.)
    def parse_heuristic(hstring):
        # Match the C# logic: default to Entropy if not recognized
        if hstring == "Scanline":
            return Model.Heuristic.SCANLINE
        elif hstring == "MRV":
            return Model.Heuristic.MRV
        else:
            return Model.Heuristic.ENTROPY

    # We gather both <overlapping> and <simpletiled> elements:
    elements = list(root.findall("simpletiled"))
    print(elements)

    for xelem in elements:
        # If "size" attribute missing, use 48 if overlapping, else 24
        default_size = 24
        size = int(xelem.get("size", default_size))
        width = int(xelem.get("width", size))
        height = int(xelem.get("height", size))
        periodic = xelem.get("periodic", "false").lower() == "true"

        heuristic_string = xelem.get("heuristic", "Entropy")
        heuristic = parse_heuristic(heuristic_string)
        
        # SimpleTiledModel-specific parameters
        subset = xelem.get("subset")  # can be None
        black_background = xelem.get("blackBackground", "false").lower() == "true"

        model = SimpleTiledModel.SimpleTiledModel(width, height, T, j_weights, j_propagator, tilecodes, heuristic=heuristic)

        # Number of screenshots to generate
        screenshots = int(xelem.get("screenshots", "2"))
        # Limit for model.Run() method (or -1 for no limit)
        limit = int(xelem.get("limit", "-1"))

        for i in range(screenshots):
            # Attempt up to 10 times to get a successful (non-contradictory) generation
            for k in range(10):
                print("> ", end="")
                seed = rng.randint(0, 2**31 - 1)
                success = model.run(seed, limit)
                if success:
                    print("DONE")
                    # Save result
                    out_img_path = os.path.join(out_dir, f"{name} {seed}.png")
                    model.save(out_img_path)

                    # If it's a SimpleTiledModel and user wants text output
                    # Check if xelem.Get("textOutput", false) => see if "textOutput" is true
                    text_output_flag = (
                        xelem.get("textOutput", "false").lower() == "true"
                    )
                    if (
                        isinstance(model, SimpleTiledModel.SimpleTiledModel)
                        and text_output_flag
                    ):
                        txt = model.text_output()
                        out_txt_path = os.path.join(out_dir, f"{name} {seed}.txt")
                        with open(out_txt_path, "w", encoding="utf-8") as f:
                            f.write(txt)
                    break
                else:
                    print("CONTRADICTION")

    elapsed_ms = int((time.time() - start) * 1000)
    print(f"time = {elapsed_ms}ms")


if __name__ == "__main__":
    main()
