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
print('acquisition terminée')

Set = preprocessing.preprocessing(Set)
print('preprocessing terminé')

try:
    mosaique = stitching.stitching(Set)
    print('assemblage terminé')
    cv2.imwrite('output/output_image.jpg', mosaique)
    cv2.imshow('img', mosaique)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except RuntimeError as e:
    print(f'[ERREUR] {e}')