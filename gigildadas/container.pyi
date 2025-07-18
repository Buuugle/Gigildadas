from typing import Sequence
import numpy as np


class GeneralSection:
    ID: int

    ut: float
    lst: float
    azimuth: float
    elevation: float
    opacity: float
    temperature: float
    integration_time: float
    parallactic_angle: float

    def __init__(self) -> None:
        ...


class PositionSection:
    ID: int

    source: str
    coordinate_system: int  # code
    equinox: float
    projection_system: int  # code
    center_lambda: float
    center_beta: float
    projection_angle: float
    lambda_offset: float
    beta_offset: float

    def __init__(self) -> None:
        ...


class SpectroSection:
    ID: int

    line: str
    rest_frequency: float
    channel_count: int
    reference_channel: float
    frequency_resolution: float
    frequency_offset: float
    velocity_resolution: float
    velocity_offset: float
    image_frequency: float
    velocity_type: int  # code
    doppler_correction: float

    def __init__(self) -> None:
        ...


class PlotSection:
    ID: int

    intensity_min: float
    intensity_max: float
    velocity_min: float
    velocity_max: float

    def __init__(self) -> None:
        ...


class SwitchSection:
    ID: int

    phase_count: int
    frequency_offsets: Sequence[float]
    times: Sequence[float]
    weights: Sequence[float]
    mode: int  # code
    lambda_offsets: Sequence[float]
    beta_offsets: Sequence[float]

    def __init__(self) -> None:
        ...


class CalibrationSection:
    ID: int

    beam_efficiency: float
    forward_efficiency: float
    gain_ratio: float
    water_content: float
    ambient_pressure: float
    ambient_temperature: float
    signal_atmosphere_temperature: float
    chopper_temperature: float
    cold_load_temperature: float
    signal_opacity: float
    image_opacity: float
    image_atmosphere_temperature: float
    receiver_temperature: float
    mode: int  # code
    factor: float
    site_elevation: float
    atmosphere_power: float
    chopper_power: float
    cold_power: float
    longitude_offset: float
    latitude_offset: float
    geographic_longitude: float
    geographic_latitude: float

    def __init__(self) -> None:
        ...


class Header:
    number: int
    version: int
    observation_date: int
    reduction_date: int
    lambda_offset: float
    beta_offset: float
    coordinate_system: int  # code
    kind: int  # code
    quality: int  # code
    position_angle: float
    scan: int
    sub_scan: int
    data_size: int

    def __init__(self) -> None:
        ...


class Container:

    def __init__(self) -> None:
        ...

    def set_input(self,
                  filename: str) -> None:
        """
        Sets the input file.
        """
        ...

    def get_size(self) -> int:
        """
        Returns the number of observations in the input file.
        """
        ...

    def get_headers(self,
                    start: int = 0,
                    end: int = 0) -> list[Header]:
        """
        Returns a list of the observations headers from index start to end (excluded).
        The default value "0" for the "end" parameter means "to the last observation in the file".
        """
        ...

    def get_data(self,
                 headers: Sequence[Header]) -> np.ndarray[np.float32]:
        """
        Returns a 2D numpy array of float corresponding to the data of each header
        """
        ...

    def get_sections[T](self,
                        headers: Sequence[Header],
                        type: type[T]) -> list[T]:
        """
        Returns a list containing the sections of each header for the given type.
        If the sections does not exists for an header, the list contains None at the header's index.
        """
        ...
