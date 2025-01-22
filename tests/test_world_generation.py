
import icland
import jax
import jax.numpy as jnp
import os
import pytest
import tempfile
from icland.world_gen.XMLReader import XMLReader, load_bitmap, save_bitmap, get_xml_attribute
from PIL import Image
from xml.etree.ElementTree import Element

# Test XML Reader
def test_xml_reader():
  xml_file = 'src/icland/world_gen/tilemap/data.xml'
  jit_model = XMLReader(xml_file)
  T, j_weights, j_propagator, tilecodes = jit_model.get_tilemap_data()
  print(f'T: {T}')
  print(f'j_weights: {j_weights}')
  print(f'j_propagator: {j_propagator}')
  print(f'tilecodes: {tilecodes}')

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

def test_save_bitmap():
    # Define test data
    width, height = 2, 2
    argb_data = [
        0xFFFF0000,  # Red
        0x8000FF00,  # Green with 50% transparency
        0x000000FF,  # Fully transparent blue
        0xFFFFFF00,  # Yellow
    ]
    
    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Call the save_bitmap function
        save_bitmap(argb_data, width, height, temp_path)

        # Open the saved image and verify its properties
        with Image.open(temp_path) as img:
            assert img.size == (width, height)
            img = img.convert("RGBA")
            saved_pixels = list(img.getdata())

            # Convert the expected ARGB data back to (R, G, B, A) tuples
            expected_pixels = [
                ((data >> 16) & 0xFF, (data >> 8) & 0xFF, data & 0xFF, (data >> 24) & 0xFF)
                for data in argb_data
            ]

            assert saved_pixels == expected_pixels
    finally:
        # Clean up the temporary file
        os.remove(temp_path)

def test_get_xml_attribute():
  elem = Element("test", attrib={
        "attr1": "42",
        "attr2": "3.14",
        "attr3": "true",
        "attr4": "False",
        "attr5": "some_text",
    })
    
  # Test integer casting
  assert get_xml_attribute(elem, "attr1", cast_type=int) == 42
  
  # Test float casting
  assert get_xml_attribute(elem, "attr2", cast_type=float) == 3.14
  
  # Test boolean casting
  assert get_xml_attribute(elem, "attr3", cast_type=bool) is True
  assert get_xml_attribute(elem, "attr4", cast_type=bool) is False
  
  # Test string (default, no casting)
  assert get_xml_attribute(elem, "attr5") == "some_text"
  
  # Test default value when attribute is missing
  assert get_xml_attribute(elem, "missing_attr", default="default_value") == "default_value"
  
  # Test default value when casting is applied
  assert get_xml_attribute(elem, "missing_attr", default=0, cast_type=int) == 0
  
  # Test generic type casting
  assert get_xml_attribute(elem, "attr5", cast_type=lambda x: x.upper()) == "SOME_TEXT"

def test_get_xml_attribute_no_casting():
  elem = Element("test", attrib={"attr": "123"})
  
  # Ensure the attribute value is returned as a string without casting
  assert get_xml_attribute(elem, "attr") == "123"

def test_get_xml_attribute_invalid_cast():
  elem = Element("test", attrib={"attr": "not_a_number"})
  
  # Test invalid cast raises a ValueError
  with pytest.raises(ValueError):
      get_xml_attribute(elem, "attr", cast_type=int)
  
  # Test invalid float cast
  with pytest.raises(ValueError):
      get_xml_attribute(elem, "attr", cast_type=float)

def test_get_xml_attribute_bool_edge_cases():
  elem = Element("test", attrib={
      "true_attr": "true",
      "false_attr": "false",
      "numeric_true": "1",
      "numeric_false": "0",
      "unexpected_value": "yes"
  })
  
  # Test boolean edge cases
  assert get_xml_attribute(elem, "true_attr", cast_type=bool) is True
  assert get_xml_attribute(elem, "false_attr", cast_type=bool) is False
  assert get_xml_attribute(elem, "numeric_true", cast_type=bool) is True
  assert get_xml_attribute(elem, "numeric_false", cast_type=bool) is False
  assert get_xml_attribute(elem, "unexpected_value", cast_type=bool) is False  # Default for unexpected values

def test_get_xml_attribute_no_attribute():
  elem = Element("test")
  
  # Test with no attribute present and no default
  assert get_xml_attribute(elem, "missing_attr") is None

  # Test with no attribute present and a default value  
  assert get_xml_attribute(elem, "missing_attr", default="default") == "default"

# Test Model
def test_model():
    pass

# Test converter
def test_converter():
  pass

# Test the entire pipeline