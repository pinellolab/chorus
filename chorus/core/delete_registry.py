import os
import atexit
import threading
import shutil
import tempfile
class DeleteOnExitRegistry:
    _instance = None
    _lock = threading.Lock()   # for thread-safe singleton creation

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DeleteOnExitRegistry, cls).__new__(cls)
                cls._instance._init_registry()
        return cls._instance

    def _init_registry(self):
        self._paths: list[tempfile.TemporaryDirectory] = []
        atexit.register(self._delete_files)

    def register(self, td: tempfile.TemporaryDirectory):
        """Register a file path to delete on program exit."""
        self._paths.append(td)

    def _delete_files(self):
        """Called automatically when the program exits."""
        print(self._paths)
        for td in list(self._paths):
            td.cleanup()