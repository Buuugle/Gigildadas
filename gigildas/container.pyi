from typing import Sequence
import numpy.typing as npt


class GeneralSection: ...


class PositionSection: ...


class SpectroSection: ...


class PlotSection: ...


class SwitchSection: ...


class CalibrationSection: ...


class Header:

    def __init__(self) -> None:
        ...


class Container:

    def __init__(self) -> None:
        ...

    def set_input(self,
                  filename: str) -> None:
        """Sets the input file and read its header"""
        ...

    def get_entry_count(self) -> int:
        """Returns the number of entries in the input file"""
        ...

    def get_headers(self,
                    start: int | None = None,
                    end: int | None = None) -> list[Header]:
        """Returns a list of the entry headers from index start to end (excluded)"""
        ...

    def get_data(self,
                 headers: Sequence[Header]) -> npt.NDArray:
        """Returns a 2D array of float representing the data of the given headers entries"""
        ...

    def get_sections[T](self,
                        headers: Sequence[Header],
                        type: type[T]) -> list[T]:
        """Returns a list of the sections of the given headers entries corresponding to the given type"""
        ...
