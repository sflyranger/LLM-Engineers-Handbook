from threading import Lock
from typing import ClassVar

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    Singleton metaclasses are classes of classes in python used to modify the behavior of any class using it as a metaclass.
    """

    _instances: ClassVar = {}

    _lock: Lock = Lock()

    """
    We now have a lock object that will be used to synchronize threads 
    during first access to Singleton.
    When lock is set that means that after the first instance of 
    the Singleton class is created no others are created even in multi threaded environments.
    Having this enforced prevents multiple writes and reads happening at the same time, 
    potentially corrupting data.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """

        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the 
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional, 
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singelton field
            # is already initialized, the thread wont create a new object.
            if cls not in cle._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            
        return cls._instances[cls]
