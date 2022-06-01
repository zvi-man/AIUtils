import time
import unittest
import threading
from Training.device_manager import DeviceManager, NoAvailableDeviceTimeoutException
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s, %(threadName)s, %(name)s, "%(message)s"',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d, %H:%M:%S')


class TestDeviceManager(object):
    _DEFAULT_TIME_OUT_SEC_DEVICE_MANAGER = 5
    _TIME_FOR_DEVICE_HOLD = 3
    _LONG_DEVICE_HOLD_TIME_SEC = 6
    _NUM_OF_DEVICES_TO_ACQUIRE_SINGLE_THREAD = 3
    _NUM_OF_DEVICES_TO_ACQUIRE_MULTIPLE_THREADS = 10

    def test_locking_single_thread(self):
        for _ in range(self._NUM_OF_DEVICES_TO_ACQUIRE_SINGLE_THREAD):
            self.acquire_device()

    def test_locking_multiple_threads(self):
        threads = []
        for _ in range(self._NUM_OF_DEVICES_TO_ACQUIRE_MULTIPLE_THREADS):
            t = threading.Thread(target=self.acquire_device, args=())
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def test_locking_timeout(self):
        try:
            self.acquire_device(device_hold_time_sec=self._LONG_DEVICE_HOLD_TIME_SEC)
        except NoAvailableDeviceTimeoutException:
            pass

    @staticmethod
    def acquire_device(timeout_sec: int = _DEFAULT_TIME_OUT_SEC_DEVICE_MANAGER,
                       device_hold_time_sec: int = _TIME_FOR_DEVICE_HOLD):
        logging.debug(f"Trying to get device")
        with DeviceManager(timeout_sec=timeout_sec) as device:
            logging.debug(f"Acquired Device {device}")
            time.sleep(device_hold_time_sec)
            logging.debug(f"Releasing Device: {device}")
