
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
  xml_file = 'src/icland/world_gen/tilemap/data.xml'
  return XMLReader(xml_file)
