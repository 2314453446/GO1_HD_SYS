from pycocotools.coco import COCO
import os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
import json


def cover_copy(src, dst):
    '''
    src和dst都必须是文件，该函数是执行覆盖操作
    '''
    if os.path.exists(dst):
        os.remove(dst)
        shutil.copy(src, dst)
    else:
        shutil.copy(src, dst)


def coco2voc(basedir='VOCdevkit/COCO_VOC', sourcedir='COCO'):
    """
    basedir:用来存放转换后数据和标注文件
    sourcedir:用来指定原始COCO数据集的存放位置
    """

    img_savepath = os.path.join(basedir, 'JPEGImages')
    ann_savepath = os.path.join(basedir, 'Annotations')
    main_path = os.path.join(basedir, "ImageSets/Main")
    for p in [basedir, img_savepath, ann_savepath, main_path]:
        if os.path.exists(p):
            shutil.rmtree(p)
            os.makedirs(p)
        else:
            os.makedirs(p)

    datasets = ['train2017', 'val2017']
    # datasets = ['val2017']

    for dataset in datasets:
        start = time.time()
        print(f"start {dataset}")
        no_ann = []  # 用来存放没有标注数据的图片的id,并将这些图片复制到results文件夹中
        not_rgb = []  # 是灰度图，同样将其保存

        annfile = 'instances_{}.json'.format(dataset)
        annpath = os.path.join(sourcedir, 'annotations', annfile)

        print('loading annotations into memory...')
        tic = time.time()
        with open(annpath, 'r') as f:
            dataset_ann = json.load(f)
        assert type(
            dataset_ann
        ) == dict, 'annotation file format {} not supported'.format(
            type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        coco = COCO(annpath)
        classes = dict()
        for cat in coco.dataset['categories']:
            classes[cat['id']] = cat['name']
        imgIds = coco.getImgIds()
        # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
        for imgId in tqdm(imgIds):
            img = coco.loadImgs(imgId)[0]

            filename = img['file_name']
            filepath = os.path.join(sourcedir, dataset, filename)

            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)

            if not len(anns):
                # print(f"{dataset}:{imgId}该文件没有标注信息，将其复制到{dataset}_noann_result中，以使查看")
                no_ann.append(imgId)
                result_path = os.path.join(sourcedir, dataset + "_noann_result")
                dest_path = os.path.join(result_path, filename)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                cover_copy(filepath, dest_path)
                continue  # 如果没有标注信息，则把没有标注信息的图片移动到相关结果文件 noann_result中,来进行查看 ，然后返回做下一张图
            # 有标注信息，接着往下走，获取标注信息
            objs = []
            for ann in anns:
                name = classes[ann['category_id']]
                if 'bbox' in ann:
                    # print('bbox in ann',imgId)
                    bbox = ann['bbox']
                    xmin = (int)(bbox[0])
                    ymin = (int)(bbox[1])
                    xmax = (int)(bbox[2] + bbox[0])
                    ymax = (int)(bbox[3] + bbox[1])
                    obj = [name, 1.0, xmin, ymin, xmax, ymax]
                    # 标错框在这里
                    if not (xmin - xmax == 0 or ymin - ymax == 0):
                        objs.append(obj)

                else:
                    print(f"{dataset}:{imgId}bbox在标注文件中不存在")  # 单张图有多个标注框，某个类别没有框

            annopath = os.path.join(ann_savepath, filename[:-3] + "xml")  # 生成的xml文件保存路径
            dst_path = os.path.join(img_savepath, filename)

            im = Image.open(filepath)
            image = np.array(im).astype(np.uint8)

            if im.mode != "RGB":

                # if img.shape[-1] != 3:

                # print(f"{dataset}:{imgId}该文件非rgb图，其复制到{dataset}_notrgb_result中，以使查看")
                # print(f"img.shape{image.shape} and img.mode{im.mode}")
                not_rgb.append(imgId)
                result_path = os.path.join(sourcedir, dataset + "_notrgb_result")
                dest_path = os.path.join(result_path, filename)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                cover_copy(filepath, dest_path)  # 复制到notrgb_result来方便查看

                im = im.convert('RGB')
                image = np.array(im).astype(np.uint8)
                im.save(dst_path, quality=95)  # 图片经过转换后，放到我们需要的位置片
                im.close()

            else:

                cover_copy(filepath, dst_path)  # 把原始图像复制到目标文件夹
            E = objectify.ElementMaker(annotate=False)
            anno_tree = E.annotation(
                E.folder('VOC'),
                E.filename(filename),
                E.source(
                    E.database('COCO'),
                    E.annotation('VOC'),
                    E.image('COCO')
                ),
                E.size(
                    E.width(image.shape[1]),
                    E.height(image.shape[0]),
                    E.depth(image.shape[2])
                ),
                E.segmented(0)
            )

            for obj in objs:
                E2 = objectify.ElementMaker(annotate=False)
                anno_tree2 = E2.object(
                    E.name(obj[0]),
                    E.pose(),
                    E.truncated("0"),
                    E.difficult(0),
                    E.bndbox(
                        E.xmin(obj[2]),
                        E.ymin(obj[3]),
                        E.xmax(obj[4]),
                        E.ymax(obj[5])
                    )
                )
                anno_tree.append(anno_tree2)
            etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
        print(f"{dataset}该数据集有{len(no_ann)}/{len(imgIds)}张图片没有instance标注信息，已经这些图片复制到{dataset}_noann_result中以使进行查看")
        print(f"{dataset}该数据集有{len(not_rgb)}/{len(imgIds)}张图片是非RGB图像，已经这些图片复制到{dataset}_notrgb_result中以使进行查看")
        duriation = time.time() - start
        print(f"数据集{dataset}处理完成用时{round(duriation / 60, 2)}分")

if __name__ == '__main__':
    """
    basedir:用来存放转换后数据和标注文件
    sourcedir:用来指定原始COCO数据集的存放位置
    """
    basedir = "/home/zhewang/Project/Dataset/voc_slices/VOC2007"
    sourcedir = "/home/zhewang/Project/Dataset/coco_sliced"
    coco2voc(basedir,sourcedir)