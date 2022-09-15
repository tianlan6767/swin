import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from glob import glob
import os.path as osp
import os, shutil, json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import mmcv, torch, cv2
from mmdet.core.mask.structures import BitmapMasks
import time


def create_dir(folder, del_existence=False):
    """
        创建指定路径并返回创建的路径

    :param folder: 需创建的路径
    :param del_existence: 是否删除已存在的文件夹
    :return: 输入的路径
    """

    if not isinstance(del_existence, bool):
        raise ValueError('del_existence is bool')

    try:
        if del_existence and os.path.exists(folder):
            shutil.rmtree(folder)
    except Exception:
        pass

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except FileExistsError:
        # TODO: 多进程或多线程调用时，需要优化
        pass

    return folder

def save_json(data, path, name, removed=False):
    """
        将字典数据保存成json文件

    :param data: 字典数据
    :param path: json文件保存文件夹
    :param name: json文件保存名称，当名称重复时会自动增加时间后缀
    :param removed: 是否移除已存在的文件
    :return: 保存的路径
    """

    if data is None or len(data) == 0:
        return

    create_dir(path)
    name = name.replace('.json', '') if name.endswith('.json') else name
    save_path = os.path.join(path, '{}.json'.format(name))
    if os.path.exists(save_path):
        if not removed:
            import time
            cur_time = time.strftime('%m%d%H%M', time.localtime(time.time()))
            save_path = os.path.join(path, '{}_{}.json'.format(name, cur_time))
        else:
            os.remove(save_path)
    print(f'{name}.json is saving...')
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print('save successfully! ->PATH: {}'.format(save_path))
    return save_path

def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes

def get_result(result, score_thr):
    # img = mmcv.imread(imp)
    # width, height = img.shape[1], img.shape[0]
    # img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        try:
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        except:
            print("not mask")
    # if out_file specified, do not show image in window
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]
        if segms is not None:
            segms = segms[inds, ...]
    nbboxes = []
    
    # bbox的值取整
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        nbboxes.append(np_poly)

    regions = []
    if not len(scores):
        return []
    # print("当前得分",scores)
    assert len(scores) == len(labels) == len(bboxes) == segms.shape[0], "not match"
    for score, label, bbox, segm in zip(scores, labels, bboxes, segms):
        res, _ = mask_to_polygons(segm)
        mask = [m.reshape(-1, 2) for m in res]
        for m in mask:
            region = {"shape_attributes":{}, "region_attributes":{}}
            xs = list(map(int, m[:,0]))
            ys = list(map(int, m[:,1]))       
            region["shape_attributes"]["all_points_x"] = xs
            region["shape_attributes"]["all_points_y"] = ys
            region["region_attributes"]["regions"] = str(label+1)
            region["region_attributes"]["score"] = str(round(score, 5))
            regions.append(region)
    return regions

def main():

    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument(
        '--config',
        default=
        r"/work/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",
        help='Config file')
    parser.add_argument(
        '--checkpoint',
        default=
        r"/work/Swin-Transformer-Object-Detection/work_dirs/run/train-all7-amp/epoch_290.pth",
        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    score_thr = 0.3

    src = r'/work/data/855G3/test/imgs'
    dst = r'/work/data/855G3/inf_test'


    # src = r"/work/data/855G/test/KL-test"
    # dst = r"/work/data/855G/test/inf_test"
    imps = glob(src + "/*.bmp")
    Path(dst).mkdir(parents=True, exist_ok=True)
        
    inf_jsd = {}
    times = []
    for imp in tqdm(imps):
        imn = osp.basename(imp)
        inf_jsd[imn] = {
                "filename":imn,
                "regions":[],
                "type":"inf"
            }
        st = time.time()
        result = inference_detector(model, imp)
        tt = round(time.time()- st, 5)
        times.append(tt)
        regions = get_result(result, score_thr)
        inf_jsd[imn]["regions"] = regions
        

        # 原始结果 show the results
        # img = show_result_pyplot(model, imp, result, score_thr=args.score_thr)
        # cv2.imwrite(osp.join(dst, imn.replace(".bmp", ".jpg")), img)
    print(times)
    print("{}張圖片推理耗時：{}".format(len(times), sum(times)/len(times)))
    save_json(inf_jsd, dst, "info-swin-03-test-290")


if __name__ == '__main__':

    main()
