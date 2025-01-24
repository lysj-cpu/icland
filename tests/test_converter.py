"""Test for our converter (from JAX arrays to STL)."""

import pytest

from icland.world_gen.XMLReader import XMLReader


@pytest.fixture
def xml_reader():
    """Fixture to create an XMLReader instance with our data XML file."""
    xml_file = "src/icland/world_gen/tilemap/data.xml"
    return XMLReader(xml_file)
