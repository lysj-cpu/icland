import pytest

from icland.world_gen.XMLReader import (
    XMLReader,
)


@pytest.fixture
def xml_reader():
    xml_file = "src/icland/world_gen/tilemap/data.xml"
    return XMLReader(xml_file)
