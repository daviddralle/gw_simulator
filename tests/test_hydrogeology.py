from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.transform import from_origin

from gw_simulator.hydrogeology import _pelletier_landform_thickness


def _write_test_raster(path: Path, values: np.ndarray, *, nodata=-1.0) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=nodata,
    ) as destination:
        destination.write(values, 1)


def test_pelletier_mosaic_uses_regolith_upland_and_sediment_lowland():
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        land_cover = np.array([[1, 2], [2, 1]], dtype="uint8")
        regolith = np.array([[30.0, -1.0], [-1.0, 40.0]], dtype="float32")
        sediment = np.array([[-1.0, 5.0], [8.0, -1.0]], dtype="float32")
        paths = {
            "land_cover": directory / "land_cover.tif",
            "regolith": directory / "regolith.tif",
            "sediment": directory / "sediment.tif",
        }
        _write_test_raster(paths["land_cover"], land_cover, nodata=255)
        _write_test_raster(paths["regolith"], regolith)
        _write_test_raster(paths["sediment"], sediment)

        thickness, classes, _, _ = _pelletier_landform_thickness(
            regolith_path=paths["regolith"],
            sediment_path=paths["sediment"],
            land_cover_path=paths["land_cover"],
            shape=(2, 2),
            transform=from_origin(0.0, 2.0, 1.0, 1.0),
            crs=rasterio.crs.CRS.from_epsg(4326),
            inside=np.ones((2, 2), dtype=bool),
        )

        np.testing.assert_array_equal(classes, land_cover)
        np.testing.assert_allclose(thickness, [[30.0, 5.0], [8.0, 40.0]])
