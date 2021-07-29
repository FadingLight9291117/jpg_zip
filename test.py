import cv2

from main import img_compose_cv2


def test_img_compose():
    img_path = 'data/imgs/0.jpg'
    img = cv2.imread(img_path)
    img_new = img_compose_cv2(img, 0.5)
    cv2.imshow('1', img)
    cv2.imshow('2', img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_img_compose()
