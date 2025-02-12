"""Test scripts for world generation part of the pipeline."""

import os
import tempfile
from pathlib import Path
from xml.etree.ElementTree import Element

import jax
import jax.numpy as jnp
import pytest
from PIL import Image

from icland.world_gen.XMLReader import (
    TileType,
    XMLReader,
    get_xml_attribute,
    load_bitmap,
    save_bitmap,
)


@pytest.fixture
def xml_reader() -> XMLReader:
    """Fixture to create an XMLReader instance with our data XML file."""
    xml_file = "src/icland/world_gen/tilemap/data.xml"
    return XMLReader(xml_file)


@pytest.fixture
def xml_reader_with_subset() -> XMLReader:
    """Fixture to create XMLReader instance with subset."""
    xml_file = "src/icland/world_gen/tilemap/data.xml"
    return XMLReader(xml_file, "two_levels")


def test_load_bitmap() -> None:
    """Test loading bitmap function for debugging tilemaps.

    This test verifies that the `load_bitmap` function correctly loads a
    bitmap from a temporary PNG file, converts the pixel data to ARGB format,
    and returns the expected width, height, and pixel data. It also ensures
    that transparency and color information are correctly interpreted.
    """
    # Create a simple 2x2 RGBA image in memory
    img = Image.new("RGBA", (2, 2), color=(255, 0, 0, 255))  # Red pixels
    img.putpixel((1, 0), (0, 255, 0, 128))  # Green with 50% transparency
    img.putpixel((0, 1), (0, 0, 255, 0))  # Blue, fully transparent
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


def test_save_bitmap() -> None:
    """Test saving bitmap function for debugging tilemaps.

    This test ensures that the `save_bitmap` function can correctly save
    ARGB pixel data to a PNG file, and that the saved image matches the
    expected dimensions and pixel data. It checks that the pixels are
    written in the expected format, including transparency.
    """
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
                (
                    (data >> 16) & 0xFF,
                    (data >> 8) & 0xFF,
                    data & 0xFF,
                    (data >> 24) & 0xFF,
                )
                for data in argb_data
            ]

            assert saved_pixels == expected_pixels
    finally:
        # Clean up the temporary file
        os.remove(temp_path)


def test_get_xml_attribute() -> None:
    """Test general behaviour of XML reader in retrieving attributes.

    This test checks if the `get_xml_attribute` function can correctly
    retrieve attributes from an XML element, with correct type casting
    for different data types such as int, float, bool, and string. It also
    verifies that default values are used when attributes are missing, and
    that generic type casting works as expected.
    """
    elem = Element(
        "test",
        attrib={
            "attr1": "42",
            "attr2": "3.14",
            "attr3": "true",
            "attr4": "False",
            "attr5": "some_text",
        },
    )

    # Test integer casting
    assert get_xml_attribute(elem, "attr1", cast_type=int) == 42

    # Test float casting
    assert get_xml_attribute(elem, "attr2", cast_type=float) == 3.14

    # Test boolean casting
    assert get_xml_attribute(elem, "attr3", cast_type=bool)
    assert not get_xml_attribute(elem, "attr4", cast_type=bool)

    # Test string (default, no casting)
    assert get_xml_attribute(elem, "attr5") == "some_text"

    # Test default value when attribute is missing
    assert (
        get_xml_attribute(elem, "missing_attr", default="default_value")
        == "default_value"
    )

    # Test default value when casting is applied
    assert get_xml_attribute(elem, "missing_attr", default=0, cast_type=int) == 0

    # Test generic type casting
    assert (
        get_xml_attribute(elem, "attr5", cast_type=lambda x: x.upper()) == "SOME_TEXT"
    )


def test_get_xml_attribute_no_casting() -> None:
    """Test behaviour of XML reader in handling valid casts."""
    elem = Element("test", attrib={"attr": "123"})

    # Ensure the attribute value is returned as a string without casting
    assert get_xml_attribute(elem, "attr") == "123"


def test_get_xml_attribute_invalid_cast() -> None:
    """Test behaviour of XML reader in handling invalid casts."""
    elem = Element("test", attrib={"attr": "not_a_number"})

    # Test invalid cast raises a ValueError
    with pytest.raises(ValueError):
        get_xml_attribute(elem, "attr", cast_type=int)

    # Test invalid float cast
    with pytest.raises(ValueError):
        get_xml_attribute(elem, "attr", cast_type=float)


def test_get_xml_attribute_bool_edge_cases() -> None:
    """Test behaviour of XML reader in edge cases."""
    elem = Element(
        "test",
        attrib={
            "true_attr": "true",
            "false_attr": "false",
            "numeric_true": "1",
            "numeric_false": "0",
            "unexpected_value": "yes",
        },
    )

    # Test boolean edge cases
    assert get_xml_attribute(elem, "true_attr", cast_type=bool)
    assert not get_xml_attribute(elem, "false_attr", cast_type=bool)
    assert get_xml_attribute(elem, "numeric_true", cast_type=bool)
    assert not get_xml_attribute(elem, "numeric_false", cast_type=bool)
    assert not (
        get_xml_attribute(elem, "unexpected_value", cast_type=bool)
    )  # Default for unexpected values


def test_get_xml_attribute_no_attribute() -> None:
    """Test behaviour of XML reader when XML has no attributes."""
    elem = Element("test")

    # Test with no attribute present and no default
    assert get_xml_attribute(elem, "missing_attr") is None

    # Test with no attribute present and a default value
    assert get_xml_attribute(elem, "missing_attr", default="default") == "default"


@pytest.mark.usefixtures("xml_reader")
def test_get_tilemap_data(xml_reader: XMLReader) -> None:
    """Test the get_tilemap_data method of the XMLReader class."""
    T, j_weights, j_propagator, j_tilecodes = xml_reader.get_tilemap_data()

    # Check the returned Tuple structure
    assert isinstance(T, int)
    assert isinstance(j_weights, jax.Array)
    assert isinstance(j_propagator, jax.Array)
    assert isinstance(j_tilecodes, jax.Array)

    # Check basic properties of the returned data
    assert T > 0  # Should have at least one tile
    assert j_weights.shape[0] == T
    assert j_propagator.shape[1] == T
    assert j_tilecodes.shape[0] == T


@pytest.mark.usefixtures("xml_reader")
def test_save(xml_reader: XMLReader, tmp_path: Path) -> None:
    """Test the save method of the XMLReader class."""
    width, height = 2, 2
    observed = jnp.array([0, 0, 0, 0])  # Mock observed array with tile indices
    filename = tmp_path / "output_tilemap.png"
    # Call the save method
    xml_reader.save(observed, width, height, str(filename))

    # Verify that the file was created
    assert os.path.exists(filename)

    # Check the file content using PIL
    with Image.open(filename) as img:
        assert img.size == (width * xml_reader.tilesize, height * xml_reader.tilesize)
        assert img.format == "PNG"


@pytest.mark.usefixtures("xml_reader")
def test_xml_reader(xml_reader: XMLReader) -> None:
    """Test the XMLReader class attributes and data integrity.

    This test verifies that the various attributes of the `XMLReader`
    class, such as `tiles`, `tilenames`, `tilecodes`, `weights`, and
    `propagator`, are correctly initialized and match the expected
    structure and data types. It checks that the number of tiles (`T`)
    corresponds to the length of these attributes.
    """
    # Attributes:
    #       tiles (list): Pixel data arrays for each tile variant.
    #       tilenames (list): Names of tiles (including variants).
    #       tilesize (int): Size (width and height) of each tile in pixels.
    #       tilecodes (list): Encoded tile properties as 4-tuples (type, rotation, from, to).
    #       weights (list): Weights associated with each tile variant.
    #       propagator (list): Sparse adjacency data indicating valid neighboring tiles.
    #       j_propagator (jax.numpy.array): JAX-compatible array representation of `propagator`.
    #       j_weights (jax.numpy.array): JAX-compatible array of tile weights.
    #       j_tilecodes (jax.numpy.array): JAX-compatible array of tile properties.
    #       T (int): Number of tile variants.

    assert xml_reader.T == len(xml_reader.tiles)
    assert xml_reader.T == len(xml_reader.tilenames)
    assert xml_reader.T == len(xml_reader.tilecodes)
    assert xml_reader.T == len(xml_reader.weights)
    assert xml_reader.T == len(xml_reader.propagator[0])
    assert xml_reader.tilesize == 8
    assert "ramp_1_2 0" in xml_reader.tilenames[0]
    assert "square_turn_6 3" in xml_reader.tilenames[-1]
    assert xml_reader.tilecodes[0] == (TileType.RAMP.value, 0, 1, 2)
    assert xml_reader.weights[0] == 3.0
    # One of the tiles that could be right of tile 0 (because 2 is left to right)
    assert xml_reader.tilenames[xml_reader.propagator[2][0][1]] == "square_boundary_1 2"
    assert len(xml_reader.propagator) == 4

    assert xml_reader.j_propagator.at[2, 0, 1].get() == xml_reader.tilenames.index(
        "square_boundary_1 2"
    )


@pytest.mark.usefixtures("xml_reader_with_subset")
def test_xml_reader_with_subset(xml_reader_with_subset: XMLReader) -> None:
    """Checks if the `T` attribute of the `xml_reader_with_subset` object is equal to a predefined number."""
    NUM_ACTIONS_SUBSET = 34
    assert xml_reader_with_subset.T == NUM_ACTIONS_SUBSET
