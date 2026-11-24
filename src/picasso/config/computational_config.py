"""Computational configuration for RIPPLE."""

from typing import Annotated

import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

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
        - A numeric string, e.g. "3", which will be converted to "cuda:3".
        - A device specifier string, e.g. "cuda:0", which means to use GPU with ID 0.
        - A list of integers, e.g. [0, 1] or in YAML format:
          gpu_ids:
            - 0
            - 1
        - A list of strings, e.g. ["cuda:0", "cuda:1"] or ["0", "1"].
        - The specific string "cpu" which means to use CPU.
        - The specific string "all" which means to use all available CUDA devices.
    """

    # Support single value, list, "all", or "cpu"
    gpu_ids: str | NonNegativeInt | list[str | NonNegativeInt] | None = Field(default=0)
    images_per_device: int | None = Field(
        default=None,
        description=(
            "Concurrent images per device. None = choose from free VRAM (GPU) "
            "or cores/RAM (CPU). A positive int forces that concurrency."
        ),
    )
    cpu_threads_per_worker: int = Field(
        default=1,
        ge=1,
        description="BLAS/PyTorch threads per CPU process worker.",
    )

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
            normalized_list = []
            for item in v:
                if not isinstance(item, (int, str)):
                    raise ValueError(
                        f"Invalid type in gpu_ids list: {type(item)}. "
                        "Expected int or str."
                    )
                # Convert numeric strings to cuda:X format
                if isinstance(item, str):
                    if item == "cpu":
                        normalized_list.append(item)
                    elif item.isdigit():
                        # Convert numeric string like "3" to "cuda:3"
                        normalized_list.append(f"cuda:{item}")
                    elif item.startswith("cuda:"):
                        normalized_list.append(item)
                    else:
                        raise ValueError(
                            f"Invalid string value in gpu_ids: {item}. "
                            "Expected 'cpu' or a cuda device specifier like 'cuda:0'."
                        )
                else:
                    normalized_list.append(item)
            return normalized_list

        # Handle single int
        if isinstance(v, int):
            return v

        # Handle single string (must be 'cpu', 'all', 'cuda:X', or a numeric string)
        if isinstance(v, str):
            if v == "cpu" or v == "all":
                return v
            elif v.isdigit():
                # Convert numeric string like "3" to "cuda:3"
                return f"cuda:{v}"
            elif v.startswith("cuda:"):
                return v
            else:
                raise ValueError(
                    f"Invalid string value for gpu_ids: {v}. "
                    "Expected 'cpu', 'all', or a cuda device specifier like 'cuda:0'."
                )

        # If we get here, the type is invalid
        raise ValueError(
            f"Invalid type for gpu_ids: {type(v)}. Expected int, str, list, or None."
        )

    @field_validator("images_per_device")  # type: ignore[misc]
    @classmethod
    def validate_images_per_device(cls, v: int | None) -> int | None:
        """Require a positive concurrency when the field is set."""
        if v is not None and v < 1:
            raise ValueError("images_per_device must be >= 1 when set.")
        return v

    @model_validator(mode="after")  # type: ignore[misc]
    def reject_mixed_cpu_gpu(self) -> Self:
        """Estimation/refine runs are GPU-only or cpu, not mixed."""
        gpu_ids = self.gpu_ids
        if not isinstance(gpu_ids, list):
            return self
        has_cpu = any(item == "cpu" for item in gpu_ids)
        has_gpu = any(item != "cpu" for item in gpu_ids)
        if has_cpu and has_gpu:
            raise ValueError(
                "gpu_ids cannot mix CPU and GPU devices. "
                "Use gpu_ids: cpu, or a list of GPU ids only."
            )
        return self

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
