import config
import acquisition
import detection
import postprocessing
import preprocessing
import stitching
import IA
import cv2
import time

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

    # Détection IA sur la mosaïque finale
    ############
    mosaique = cv2.imread('carviewalive.jpg')
    ############
    detections = IA.detect(mosaique)
    print(f"Détections : {len(detections)} voiture(s)")
    for d in detections:
        print(f"  box={[round(v,1) for v in d['box']]}  score={d['score']:.2f}")
    pred_tensor = model(preprocess(mosaique).unsqueeze(0))[0].cpu()
    IA.plot_prediction(mosaique)
    detection.pad_to_square(mosaique)
    print("Full process finished")

    end_time = time.perf_counter()
    print(f"Duration : {end_time - start_time:.6f} s")

except RuntimeError as e:
    print(f'[ERROR] {e}')