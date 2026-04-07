import config
import acquisition
import detection
import postprocessing
import preprocessing
import stitching
import cv2
import time

# Test 1

Set = acquisition.acquisition()
print('acquisition finished')

Set = preprocessing.preprocessing(Set)
print('preprocessing finished')

try:
    start_time = time.perf_counter()
    mosaique = stitching.stitching(Set)
    print('assemblage finished')
    mosaique = postprocessing.postprocessing(mosaique)
    cv2.imwrite('output/output_image.jpg', mosaique)
    print("Full process finished")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Duration : {duration:.6f} s")

except RuntimeError as e:
    print(f'[ERROR] {e}')