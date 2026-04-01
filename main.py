import config
import acquisition
import detection
import postprocessing
import preprocessing
import stitching
import cv2
import test # TEST

# Test 1
Set = acquisition.acquisition()
print('acquisition finished')

Set = preprocessing.preprocessing(Set)
print('preprocessing finished')

try:
    mosaique = stitching.stitching(Set)
    print('assemblage finished')
    mosaique = postprocessing.postprocessing(mosaique)
    cv2.imwrite('output/output_image.jpg', mosaique)
    print("Full process finished")
except RuntimeError as e:
    print(f'[ERROR] {e}')