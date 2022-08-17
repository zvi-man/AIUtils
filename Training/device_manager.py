import threading
from typing import Union, Optional, Any
import torch
import time

# Constants
# NUM_OF_DEVICES = 5
NUM_OF_DEVICES = torch.cuda.device_count()
DEVICE_AVAILABLE_LIST = [True for _ in range(NUM_OF_DEVICES)]
DEVICE_LIST_LOCK = threading.Lock()
GET_AVAILABLE_DEVICE_TIMEOUT_SEC = 5
AVAILABLE_DEVICE_POOLING_INTERVAL_SEC = 1
CUDA_STR = "cuda:"


class NoAvailableDeviceTimeoutException(Exception):
    pass


class DeviceManagerException(Exception):
    pass


class DeviceManager(object):
    def __init__(self, timeout_sec: Union[int, None] = GET_AVAILABLE_DEVICE_TIMEOUT_SEC):
        self.timeout_sec = timeout_sec
        self.device_id: Optional[int] = None

    def __enter__(self) -> torch.device:
        self.device_id = self.acquire_device_timeout(self.timeout_sec)
        return torch.device(CUDA_STR + str(self.device_id))

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        assert self.device_id is not None
        self.release_device(self.device_id)

    def acquire_device_timeout(self, timeout_sec: Union[int, None]) -> int:
        start_time = time.time()
        while timeout_sec is None or time.time() < (start_time + timeout_sec):
            with DEVICE_LIST_LOCK:
                if self.is_device_available():
                    available_device_id = DEVICE_AVAILABLE_LIST.index(True)
                    self.acquire_device(available_device_id)
                    return available_device_id
            time.sleep(AVAILABLE_DEVICE_POOLING_INTERVAL_SEC)
        raise NoAvailableDeviceTimeoutException("Timeout expired - Could not find available device")

    @staticmethod
    def acquire_device(device_id: int) -> None:
        if not DEVICE_AVAILABLE_LIST[device_id]:
            raise DeviceManagerException(f"Cannot set device_id: {device_id} to not available. "
                                         f"device already unavailable")
        DEVICE_AVAILABLE_LIST[device_id] = False

    @staticmethod
    def release_device(device_id: int) -> None:
        with DEVICE_LIST_LOCK:
            if DEVICE_AVAILABLE_LIST[device_id]:
                raise DeviceManagerException(f"Cannot set device_id: {device_id} to available. "
                                             f"device already available")
            DEVICE_AVAILABLE_LIST[device_id] = True

    @staticmethod
    def is_device_available() -> bool:
        return any(DEVICE_AVAILABLE_LIST)
