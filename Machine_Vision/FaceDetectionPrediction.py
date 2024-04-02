import mediapipe as mp
import cv2 as cv
import numpy as np
import tensorflow as tf


def main():


    def get_bounding_rectangle(image,region, color, radius):

        height, width = image.shape[0:2]

        bounding_box = region.location_data.relative_bounding_box

        left = bounding_box.xmin
        top = bounding_box.ymin

        right = left + bounding_box.width
        bottom = top + bounding_box.height

        left = (int)(left * width)
        top = (int)(top * height)

        if left <0:
            left =0
        if top <0:
            top = 0

        right = (int)(right * width)
        bottom = (int)(bottom * height)



        return left, top, right, bottom


    def drw_bounding_rectangle(left, top, right, bottom, image, color, color2, color3, str_Mask, str_Marks2, str_Marks3, str_Score, str_Score2, str_Score3):

        cv.rectangle(image, (left, top), (right, bottom), color, 1)
        cv.rectangle(image, (left, top), (right, bottom), color2, 1)
        cv.rectangle(image, (left, top), (right, bottom), color3, 1)



        cv.line(image, (left, top), (left, top+10), color, 3)
        cv.line(image, (left, top), (left+10, top), color, 3)

        cv.line(image, (right, top), (right, top+10), color, 3)
        cv.line(image, (right, top), (right-10, top), color, 3)


        cv.line(image, (left, bottom), (left, bottom-10), color, 3)
        cv.line(image, (left, bottom), (left+10, bottom), color, 3)

        cv.line(image, (right, bottom), (right, bottom-10), color, 3)
        cv.line(image, (right, bottom), (right-10, bottom), color, 3)

        cv.putText(image, str_Mask, (left, top-5), cv.FONT_HERSHEY_SIMPLEX, 0.7,  color,  2)
        cv.putText(image, str_Score, (right, top-5), cv.FONT_HERSHEY_SIMPLEX, 0.7,  color,  2)
        cv.putText(image, str_Marks2, (left, top - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.putText(image, str_Score2, (right, top - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.putText(image, str_Marks3, (left, top - 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv.putText(image, str_Score3, (right, top - 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    def convert_img_np(image_resize, target_size):

        image_s = np.array(image_resize, dtype=object)
        image_s = image_s.reshape(1, target_size, target_size, 3)
        image_s = image_s.astype('float32')
        image_s /= 255

        return image_s



    COLOR_GREEN = (0,255,0)
    COLOR_RED = (0,0,255)
    COLOR_BLUE = (255,0,0)


    face_detection = mp.solutions.face_detection

    capture = cv.VideoCapture(0)


    MODEL_SELECTION = 1
    CONFIDENCE = 0.5

    TARGET_SIZE = 256

    detection = face_detection.FaceDetection(model_selection=MODEL_SELECTION,
                                             min_detection_confidence= CONFIDENCE)

    model_path = "FaceMaskGPU.h5"
    model_path2 = "FaceMaskGPU2.h5"
    model_path3 = "FaceMaskGPU3.h5"

    model = tf.keras.models.load_model(model_path)
    model2 = tf.keras.models.load_model(model_path2)
    model3 = tf.keras.models.load_model(model_path3)


    while True:
        results, image = capture.read()

        if results:

            image_convert = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            outputs = detection.process(image_convert)

            if outputs.detections:

                 for region in outputs.detections:

                    left, top, right, bottom = get_bounding_rectangle(image,region, COLOR_GREEN, 2)

                    crop_image = image[top:bottom, left:right]


                    image_resize = cv.resize(crop_image, (TARGET_SIZE, TARGET_SIZE))

                    image_s = convert_img_np(image_resize, TARGET_SIZE)

                    # detect_class = model.predict_classes(image_s)
                    #
                    r = model.predict(image_s)
                    a = np.max(r)
                    r2 = model2.predict(image_s)
                    a2 = np.max(r2)
                    r3 = model3.predict(image_s)
                    a3 = np.max(r3)

                    class_index = np.where(r==a)
                    class_index2 = np.where(r2==a2)
                    class_index3 = np.where(r3 == a3)

                    detect_class = class_index[1]
                    detect_class2 = class_index2[1]
                    detect_class3 = class_index3[1]

                    score = r[0][0]
                    score = float("{:.2f}".format(score))
                    score2 = r2[0][0]
                    score2 = float("{:.2f}".format(score2))
                    score3 = r3[0][0]
                    score3 = float("{:.2f}".format(score3))

                    if detect_class == 0:
                        color = COLOR_GREEN
                        str_Mask = 'WITH MASK'
                    else:
                        color = COLOR_RED
                        str_Mask = 'NO MASK'

                    if detect_class2 == 0:
                        color2 = COLOR_GREEN
                        str_Mask2 = 'WITH GLASSES'
                    else:
                        color2 = COLOR_RED
                        str_Mask2 = 'NO GLASSES'

                    if detect_class3 == 0:
                        color3 = COLOR_GREEN
                        str_Mask3 = 'WITH Doctor CAP'
                    else:
                        color3 = COLOR_RED
                        str_Mask3 = 'NO Doctor CAP'

                    drw_bounding_rectangle(left, top, right, bottom, image, color, color2, color3,str_Mask, str_Mask2, str_Mask3, str(score), str(score2), str(score3))
                    # drw_bounding_rectangle(left, top, right, bottom, image, color2, str_Mask2,str(score2))



            cv.imshow("Face_Detection", image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break

    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()