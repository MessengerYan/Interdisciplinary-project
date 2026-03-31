import config
import acquisition
import detection
import orthorectification
import postprocessing
import preprocessing
import stitching
import cv2

# Test 1
Set = acquisition.acquisition()
print('acquisition finished')

Set = preprocessing.preprocessing(Set)
print('preprocessing finished')

try:
    mosaique = stitching.stitching(Set)
    print('assemblage finished')
    cv2.imwrite('output/output_image.jpg', mosaique)
    cv2.imshow('img', mosaique)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except RuntimeError as e:
    print(f'[ERROR] {e}')