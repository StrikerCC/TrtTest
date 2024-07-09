import cv2


def main():
    res_img_fp = './cmake-build-release-visual-studio/seg.png'
    img = cv2.imread(res_img_fp)

    img = img * 70

    # cv2.namedWindow('res', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('seg', img)
    cv2.namedWindow('seg', cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    cv2.imwrite('./cmake-build-release-visual-studio/segx70.png', img)
    return


if __name__ == '__main__':
    main()
