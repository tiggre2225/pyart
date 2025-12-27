"""
Reading of JRC (Japan Radio Co., Ltd) JMA-254 S-band raw radar files.

"""

import datetime

import numpy as np

from ..config import FileMetadata, get_fillvalue
from ..core.radar import Radar
from .common import _test_arguments, make_time_unit_str, prepare_for_read

_JRC_PRODUCT_CODES = {
    0x75: "velocity",
    0x76: "spectrum_width",
    0xF1: "reflectivity",
}

_JRC_SCALING = {
    0x75: {"scale":  0.25, "offset":  -32.0, "missing_raw": 255},
    0x76: {"scale": 0.25, "offset": 0.0, "missing_raw": 0},
    0xF1: {"scale": 0.3125, "offset": 0.0, "missing_raw": 255},
}


def _datetime_to_float(dts, base_dt):
    """
    Convert a list of datetime objects to float seconds since base_dt.

    Parameters
    ----------
    dts : list of datetime
        List of datetime objects to convert.
    base_dt :  datetime
        Reference datetime for the conversion.

    Returns
    -------
    times : ndarray
        Array of float values representing seconds since base_dt.

    """
    return np.array(
        [(dt - base_dt).total_seconds() for dt in dts], dtype="float32"
    )


def read_jrc(
    filename,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    **kwargs,
):
    """
    Read a JRC (Japan Radio Co., Ltd) JMA-254 S-band raw radar file.

    Parameters
    ----------
    filename : str or file-like
        Name of JRC JMA-254 raw file to read data from.
    field_names :  dict, optional
        Dictionary mapping JRC data type names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata from during this read.
        This metadata is not used during any successive file reads unless
        explicitly included. A value of None, the default, will not
        introduce any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    file_field_names :  bool, optional
        True to force the use of the field names from the file in which
        case the `field_names` parameter is ignored.False will use to
        `field_names` parameter to rename fields.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object.This is applied
        after the `file_field_names` and `field_names` parameters.Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object.This is applied
        after the `file_field_names` and `field_names` parameters.Set
        to None to include all fields not specified by exclude_fields.

    Returns
    -------
    radar : Radar
        Radar object containing data from the JRC file.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    # create metadata retrieval object
    filemetadata = FileMetadata(
        "jrc",
        field_names,
        additional_metadata,
        file_field_names,
        exclude_fields,
        include_fields,
    )

    # read the JRC file
    jrc_file = JRCFile(prepare_for_read(filename))

    # time
    dts = jrc_file.get_datetimes()
    base_dt = dts[0]
    units = make_time_unit_str(base_dt)
    time = filemetadata("time")
    time["units"] = units
    time["data"] = _datetime_to_float(dts, base_dt)

    # range
    _range = filemetadata("range")
    ngates = jrc_file.ngates
    # JRC files use 1000m gate spacing, starting at 125m (center of first gate)
    gate_spacing = 1000.0  # meters
    start = 125.0  # meters to center of first gate
    _range["data"] = np.arange(ngates, dtype="float32") * gate_spacing + start
    _range["meters_to_center_of_first_gate"] = start
    _range["meters_between_gates"] = gate_spacing

    # latitude, longitude and altitude
    latitude = filemetadata("latitude")
    longitude = filemetadata("longitude")
    altitude = filemetadata("altitude")

    # Use radar location from file if available, otherwise use defaults
    lat, lon, alt = jrc_file.get_location()
    latitude["data"] = np.array([lat], dtype="float64")
    longitude["data"] = np.array([lon], dtype="float64")
    altitude["data"] = np.array([alt], dtype="float64")

    # metadata
    metadata = filemetadata("metadata")
    metadata["original_container"] = "JRC"
    metadata["instrument_name"] = jrc_file.get_instrument_name()

    # sweep_start_ray_index, sweep_end_ray_index
    sweep_start_ray_index = filemetadata("sweep_start_ray_index")
    sweep_end_ray_index = filemetadata("sweep_end_ray_index")
    nrays = jrc_file.nrays
    sweep_start_ray_index["data"] = np.array([0], dtype="int32")
    sweep_end_ray_index["data"] = np.array([nrays - 1], dtype="int32")

    # sweep number
    sweep_number = filemetadata("sweep_number")
    sweep_number["data"] = np.array([0], dtype="int32")

    # scan_type - JRC files are typically PPI surveillance scans
    scan_type = "ppi"

    # sweep_mode
    sweep_mode = filemetadata("sweep_mode")
    sweep_mode["data"] = np.array(["azimuth_surveillance"], dtype="S")

    # elevation, azimuth
    elevation = filemetadata("elevation")
    elev_angle = jrc_file.get_elevation()
    elevation["data"] = np.full(nrays, elev_angle, dtype="float32")
    azimuth = filemetadata("azimuth")
    azimuth["data"] = np.linspace(1, 360, nrays, dtype="float32")

    # fixed_angle
    fixed_angle = filemetadata("fixed_angle")
    fixed_angle["data"] = np.array([elev_angle], dtype="float32")

    # fields
    fields = {}
    jrc_field_name = jrc_file.get_field_name()
    field_name = filemetadata.get_field_name(jrc_field_name)
    if field_name is not None: 
        field_dic = filemetadata(field_name)
        field_dic["data"] = jrc_file.get_field_data()
        field_dic["_FillValue"] = get_fillvalue()
        fields[field_name] = field_dic

    jrc_file.close()

    return Radar(
        time,
        _range,
        fields,
        metadata,
        scan_type,
        latitude,
        longitude,
        altitude,
        sweep_number,
        sweep_mode,
        fixed_angle,
        sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth,
        elevation,
    )


class JRCFile:
    """
    A class for reading JRC JMA-254 S-band radar files.

    Parameters
    ----------
    fh :  file-like
        File-like object from which to read the JRC data.

    Attributes
    ----------
    nrays : int
        Number of rays in the file (always 360 for JRC files).
    ngates : int
        Number of gates per ray.
    header : ndarray
        Raw header data from the file.
    data : ndarray
        Raw data array from the file.

    """

    def __init__(self, fh):
        """Initialize a JRCFile object."""
        self._fh = fh
        self._read_file()

    def _read_file(self):
        """Read the JRC file and parse its contents."""
        # Read entire file as unsigned bytes
        raw = np.fromfile(self._fh, dtype="B")

        # Parse header (bytes 96-352)
        self.header = raw[96:352]

        # Parse data section (starts at byte 352)
        arr = raw[352:]

        # Data is arranged as 360 rays (one per degree)
        self.nrays = 360
        ncols = arr.size // self.nrays
        data = arr.reshape((self.nrays, ncols))

        # First 32 bytes of each ray is sweep header
        self._sweep_header = data[: , : 32]
        self._raw_data = data[:, 32:]
        self.ngates = self._raw_data.shape[1]

        # Determine product type and apply scaling
        self._product_code = self.header[3]
        self._apply_scaling()

    def _apply_scaling(self):
        """Apply scaling factors based on product type."""
        if self._product_code in _JRC_SCALING:
            scaling = _JRC_SCALING[self._product_code]
            scaled_data = self._raw_data * scaling["scale"] + scaling["offset"]
            missing_scaled = scaling["missing_raw"] * scaling["scale"] + scaling["offset"]
            # Replace missing values with fill value
            self._data = np.where(
                scaled_data == missing_scaled, get_fillvalue(), scaled_data
            )
            self._data_type = _JRC_PRODUCT_CODES.get(self._product_code, "unknown")
        else:
            # Unknown product type, return raw data
            self._data = self._raw_data.astype("float32")
            self._data_type = "unknown"

    def get_datetimes(self):
        """
        Return a list of datetime objects for each ray.

        Returns
        -------
        dts : list
            List of datetime objects, one for each ray.

        """
        # Extract timestamp from header (bytes 8-24)
        date_bytes = self.header[8:24].tobytes()
        timestamp_str = date_bytes.decode("ascii")
        base_dt = datetime.datetime.strptime(timestamp_str, "%Y.%m.%d.%H.%M")

        # All rays in a JRC file have the same timestamp
        return [base_dt] * self.nrays

    def get_elevation(self):
        """
        Return the elevation angle in degrees.

        Returns
        -------
        elevation :  float
            Elevation angle in degrees.

        """
        # Elevation is stored in header byte 44, scaled by 360/(2^16)
        return self.header[44] * 360.0 / (2**16)

    def get_field_name(self):
        """
        Return the field name based on product code.

        Returns
        -------
        field_name : str
            The name of the field (e.g., 'reflectivity', 'velocity').

        """
        return _JRC_PRODUCT_CODES.get(self._product_code, "unknown")

    def get_field_data(self):
        """
        Return the field data as a masked array.

        Returns
        -------
        data : MaskedArray
            Masked array containing the field data with missing values masked.

        """
        return np.ma.masked_values(self._data, get_fillvalue()).astype("float32")

    def get_location(self):
        """
        Return the radar location.

        Returns
        -------
        lat : float
            Latitude in degrees.
        lon :  float
            Longitude in degrees.
        alt : float
            Altitude in meters.

        Notes
        -----
        If location information is not available in the file, default values
        are returned. Users should override these with actual radar location.

        """
        # JRC files may not contain location info in a standard way
        # Default values - users should override as needed
        return 0.0, 0.0, 0.0

    def get_instrument_name(self):
        """
        Return the instrument name.

        Returns
        -------
        name : str
            Instrument name extracted from header or default.

        """
        # Location code is in header byte 1
        loc_code = self.header[1]
        return f"JRC_LOC{loc_code: 02d}"

    def get_step_number(self):
        """
        Return the step number from the header.

        Returns
        -------
        step :  int
            Step number from the header.

        """
        return int(self.header[42])

    def close(self):
        """Close the file handle."""
        if hasattr(self._fh, "close"):
            self._fh.close()
