
import icland
import jax
import jax.numpy as jnp
import os
import pytest
import tempfile
from icland.world_gen.XMLReader import XMLReader, TileType, load_bitmap, save_bitmap, get_xml_attribute
from PIL import Image
from xml.etree.ElementTree import Element

@pytest.fixture
def xml_reader():
  reader = XMLReader()
  t, w, p, c = reader.get_tilemap_data()
  model = init(10, 10, t, 1, False, 1, w, p, jax.random.key(0))


def test_load_bitmap():
  # Create a simple 2x2 RGBA image in memory
  img = Image.new("RGBA", (2, 2), color=(255, 0, 0, 255))  # Red pixels
  img.putpixel((1, 0), (0, 255, 0, 128))  # Green with 50% transparency
  img.putpixel((0, 1), (0, 0, 255, 0))    # Blue, fully transparent
  img.putpixel((1, 1), (255, 255, 0, 255))  # Yellow, fully opaque

  # Save to a temporary file
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
      temp_path = temp_file.name
  img.save(temp_path)

  try:
      # Test the function
      pixels, width, height = load_bitmap(temp_path)

      # Expected ARGB values
      expected_pixels = [
          0xFFFF0000,  # Red
          0x8000FF00,  # Green with 50% transparency
          0x000000FF,  # Fully transparent blue
          0xFFFFFF00,  # Yellow
      ]
      assert width == 2
      assert height == 2
      assert pixels == expected_pixels
  finally:
      # Clean up the temporary file
      os.remove(temp_path)