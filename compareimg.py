import cv2
import numpy as np
#Программма для сравнения двух изображений методом ORB+RANSAC

#загружаем изображения
img1 = cv2.imread('c:/test/test_pic_5.png',cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('c:/test/test_pic_10.png',cv2.COLOR_BGR2GRAY)

#Инициируем ORB детектор
orb = cv2.ORB_create(nfeatures=100000)

#Находим ключевые точки и дескрипторы изображения
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2,None)

#печатаем первоначальное количество ключевых точек
print('kp1 = ', len(kp1), '   kp2 = ', len(kp2))

#объект для последующего сравнения методом грубой силы
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#сопоставляем дескрипторы
matches = bf.match(des1,des2)

#сортируем их по возрастанию дистанции
matches = sorted(matches, key = lambda x:x.distance)

print ('matches after ORB = ', len(matches))

##извлекаем сопоставленные ключевые точки
src_pts  = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

#находим матрицу гомографии и делаем трансформации перспективы
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,14.0)#,maxIters=5000)


good_matches = []
counter = 0
for i in mask:
	if i!= 0:
		good_matches.append(matches[counter])
		counter+=1
print('matches after RANSAC = ', counter)


#отрисовка линий после применения ORB
#res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:],None,flags=2)
#res = cv2.resize(res, (0, 0), fx=0.5, fy=0.5)
#cv2.imshow("orb_match", res);

#отрисовка линий после применения RANSAC
res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:],None,flags=2)
res = cv2.resize(res, (0, 0), fx=0.6, fy=0.6)
cv2.imshow("good_match", res);

print('Изображения похожи на : ' + str(int(len(good_matches)/len(matches) *100)) + ' %')

cv2.waitKey()
cv2.destroyAllWindows()
