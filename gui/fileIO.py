import numpy as np
import io
import zipfile

def create_zip_download(results_dict):
    """
    results_dict: {'saxs_1d': array, 'ttc_matrix': array, ...}
    """
    # 1. Create an in-memory byte stream for the zip file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for filename, array in results_dict.items():
            # 2. Create an in-memory byte stream for the numpy array
            array_buffer = io.BytesIO()
            np.save(array_buffer, array)
            
            # 3. Write the numpy buffer into the zip file
            zip_file.writestr(f"{filename}.npy", array_buffer.getvalue())

    return zip_buffer.getvalue()


