import os
import inspect
import warnings

class CustomLogger:
    def __init__(self, command=True):
        self.command = command

    def get_file_name_without_extension(self, file_path):
        file_name = os.path.basename(file_path)
        return os.path.splitext(file_name)[0]

    def commandline(self, *args, **kwargs):
        if self.command:
            # Get the caller's file name automatically
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            file_name_without_extension = self.get_file_name_without_extension(caller_file)

            # ANSI escape code for red color
            color_code = "\033[91m"
            reset_code = "\033[0m"
            print(f"{color_code}{file_name_without_extension}:{reset_code}", *args, **kwargs)

    def raise_warning(self, message):
        if self.command:
            # Get caller details
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            file_name_without_extension = self.get_file_name_without_extension(caller_file)

            # Define a custom format for the warning message
            def custom_formatwarning(msg, *args, **kwargs):
                return f"{msg}\n"

            # Set the custom format for warnings
            warnings.formatwarning = custom_formatwarning

            # Issue the warning
            warnings.warn(f"{file_name_without_extension}: {message}", UserWarning)

    def raise_error(self, message):
        if self.command:
            # Get caller details
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            file_name_without_extension = self.get_file_name_without_extension(caller_file)

            # Raise an error
            raise Exception(f"{file_name_without_extension}: {message}")
