#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//функция для сравнения картинок(детектор и мэтчер передаются в параметрах, чтобы в функции не пересоздавать их на каждой итерации)
int CompareImages(String img1path, String img2path, Ptr<FeatureDetector> detector, BFMatcher matcher)
{
	//открываем изображения с последующей проверкой 
	Mat img1 = imread(img1path, IMREAD_GRAYSCALE);
	Mat img2 = imread(img2path, IMREAD_GRAYSCALE);
	if (img1.empty() or img2.empty())
	{
		std::cout << "Не удалось открыть изображение" << std::endl;
		return -1;
	}

	// Находим дескрипторы и ключевые точки при помощи ORB
	vector<KeyPoint> keypoints_1, keypoints_2;
	detector->detect(img1, keypoints_1);
	detector->detect(img2, keypoints_2);

	Mat descriptor_1, descriptor_2;
	detector->compute(img1, keypoints_1, descriptor_1);
	detector->compute(img2, keypoints_2, descriptor_2);

	//вектор для хранения совпадений
	vector<DMatch> matches;

	//применяем matcher к дескрипторам и получаем совпадения в matches
	matcher.match(descriptor_1, descriptor_2, matches);

	if (matches.size() == 0)
	{
		cout << "Количество совпадений == 0";
		return -1;
	}

	//вектора для хранения совпадений, полученных из ключевых точек
	std::vector<Point2f> match_pts1;
	std::vector<Point2f> match_pts2;
	//получаем ключевые точки из совпадений
	for (size_t i = 0; i < matches.size(); i++)
	{
		match_pts1.push_back(keypoints_1[matches[i].queryIdx].pt);
		match_pts2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	//отсеиваем часть точек при помощи RANSAC
	Mat mask;
	Mat H = findHomography(match_pts1, match_pts2, RANSAC, 14.0, mask);

	//считаем количество совпадений после применения RANSAC
	int counter = 0;
	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.data[i] != 0)
			counter++;
	}
	//вычисляем значение похожести = совпадения после применения RANSAC/совпадения до RANSAC
	return int((float(counter)/float(matches.size()))*100);
}


int main()
{
	//устанавливаем русскую локаль
	setlocale(LC_ALL, "Russian");

	// инициализация и считывание порогового значения "похожести" в процентах
	int similarity_limit;
	cout << "Введите пороговое значение похожести в процентах" << endl;
	cin >> similarity_limit;

	//считываем директории к файлам с изображениями, их должно быть >= 2
	vector<String> img_paths;
	String img_path;
	cout << "Введите пути к файлам изображений, для остановки CTRL+Z" << endl;
	while (cin >> img_path)
	{
		img_paths.push_back(img_path);
	}

	//проверка количества файлов, если < 2 - завершение работы
	if (img_paths.size() < 2)
	{
		cout << "Недостаточное количество файлов";
		return 1;
	}

	//Инициируем ORB детектор
	Ptr<FeatureDetector> detector = ORB::create(10000);

	//объект для дальнейшего рассчета ближайших точек методом "грубой силы"
	BFMatcher matcher(NORM_L2, true);

	//цикл перебора файлов
	for (int i = 0; i < img_paths.size()-1; i++)
		for (int j = i+1; j < img_paths.size(); j++)
		{
			//сравниваем изображения
			int compare_result = CompareImages(img_paths[i], img_paths[j], detector, matcher);
			//сверка с коэффициентом похожести и вывод результатов
			if (compare_result > similarity_limit)
				cout << img_paths[i] << " , " << img_paths[j] << " , " << compare_result << " %" << endl;
		}
	return 0;
}