import cv2 

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()

#cmake -D CMAKE_BUILD_TYPE=RELEASE \
#	-D CMAKE_INSTALL_PREFIX=/usr/local \
#	-D INSTALL_PYTHON_EXAMPLES=ON \
#	-D INSTALL_C_EXAMPLES=OFF \
#	-D OPENCV_ENABLE_NONFREE=ON \
#	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
#	-D PYTHON_EXECUTABLE=~/.virtualenvs/gaze/bin/python \
#	-D BUILD_EXAMPLES=ON \
 #   -D BUILD_LIBPROTOBUF_FROM_SOURCES=ON \
  #  -D BUILD_opencv_dnn=OFF \ -D ENABLE_PRECOMPILED_HEADERS=OFF ..


