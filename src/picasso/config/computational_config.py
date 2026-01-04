"""Computational configuration for RIPPLE."""

from typing import Annotated

import torch
from pydantic import BaseModel, Field, field_validator

# Type alias for non-negative integer
NonNegativeInt = Annotated[int, Field(ge=0)]


class ComputationalConfig(BaseModel):
    """Serialization of computational resources allocated for PICASSO.

    Attributes
    ----------
    gpu_ids : Optional[Union[int, str, list[Union[int, str]]]]
        Field which specifies which GPU(s) or CPU to use for computation.
        The following types of values are allowed:
        - A single integer, e.g. 0, which means to use GPU with ID 0.
        - A device specifier string, e.g. "cuda:0", which means to use GPU with ID 0.
        - A list of integers or strings, e.g. [0, 1] or ["cuda:0", "cuda:1"].
        - The specific string "cpu" which means to use CPU.
        - The specific string "all" which means to use all available CUDA devices.
    """

    # Support single value, list, "all", or "cpu"
    gpu_ids: str | NonNegativeInt | list[str | NonNegativeInt] | None = Field(default=0)

    @field_validator("gpu_ids")  # type: ignore[misc]
    @classmethod
    def validate_gpu_ids(
        cls, v: str | int | list[str | int] | None
    ) -> str | int | list[str | int] | None:
        """Validate that gpu_ids is valid.

        Parameters
        ----------
        v : Union[str, int, list]
            The gpu_ids value to validate.

        Returns
        -------
        Union[str, int, list]
            The validated gpu_ids value.

        Raises
        ------
        ValueError
            If invalid values are provided.
        """
        # Handle None
        if v is None:
            return v

        # Handle "all" or "cpu" strings
        if v in ("all", "cpu"):
            return v

        # Handle lists
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("gpu_ids list cannot be empty.")
            for item in v:
                if not isinstance(item, (int, str)):
                    raise ValueError(
                        f"Invalid type in gpu_ids list: {type(item)}. "
                        "Expected int or str."
                    )
                if (
                    isinstance(item, str)
                    and item != "cpu"
                    and not item.startswith("cuda:")
                ):
                    raise ValueError(
                        f"Invalid string value in gpu_ids: {item}. "
                        "Expected 'cpu' or a cuda device specifier like 'cuda:0'."
                    )
            return v

        # Handle single int
        if isinstance(v, int):
            return v

        # Handle single string (must be 'cpu' or 'cuda:X')
        if isinstance(v, str):
            if v != "cpu" and not v.startswith("cuda:"):
                raise ValueError(
                    f"Invalid string value for gpu_ids: {v}. "
                    "Expected 'cpu', 'all', or a cuda device specifier like 'cuda:0'."
                )
            return v

        # If we get here, the type is invalid
        raise ValueError(
            f"Invalid type for gpu_ids: {type(v)}. Expected int, str, list, or None."
        )

    @property
    def gpu_device(self) -> torch.device:
        """Convert requested GPU ID or CPU to a torch device object.

        Returns
        -------
        torch.device
            A single device (either GPU or CPU). For backward compatibility.
            If multiple devices are specified, returns the first one.
        """
        devices = self.gpu_devices
        return devices[0]

    @property
    def gpu_devices(self) -> list[torch.device]:
        """Convert requested GPU IDs to torch device objects.

        Returns
        -------
        list[torch.device]
            List of torch devices.
        """
        # Handle special string cases first
        if self.gpu_ids == "all":
            if not torch.cuda.is_available():
                raise ValueError("No CUDA devices available.")
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        if self.gpu_ids == "cpu":
            return [torch.device("cpu")]

        # Normalize to list for uniform processing
        # At this point, gpu_ids is not None, "all", or "cpu"
        if isinstance(self.gpu_ids, list):
            gpu_list: list[str | int] = self.gpu_ids
        else:
            # Single value case - ensure it's not None
            if self.gpu_ids is None:
                raise ValueError("gpu_ids cannot be None at this point")
            gpu_list = [self.gpu_ids]

        # Process each item in the normalized list
        devices = []
        for gpu_id in gpu_list:
            if isinstance(gpu_id, int):
                devices.append(torch.device(f"cuda:{gpu_id}"))
            elif isinstance(gpu_id, str):
                devices.append(torch.device(gpu_id))
            else:
                raise TypeError(
                    f"Invalid type for gpu_ids element: {type(gpu_id)}. "
                    "Expected int or str."
                )

        return devices
